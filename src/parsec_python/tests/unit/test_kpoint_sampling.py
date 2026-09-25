"""Tests for periodic k-point sampling (real, spin-unpolarized, no SOC).

Validates the new machinery (Laplacian.build_gradient,
V_ion.build_nonlocal_projectors's k_point argument,
Hamiltonian.kpoint_operator.KPointKohnShamHamiltonian, SCF.kpoints) against
independent analytic and reduction checks rather than only against itself.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from parsec_python import (
    Atom,
    EigensolverSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.Grid import monkhorst_pack_grid
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Laplacian import build_gradient, build_negative_laplacian
from parsec_python.SCF.kpoints import run_scf_kpoints
from parsec_python.SCF.pbc import prepare_periodic_single_point
from parsec_python.SCF.single_point import run_scf as run_scf_gamma

_AU_POTRE = (
    Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
)


class MonkhorstPackGridTests(unittest.TestCase):
    def test_1x1x1_is_gamma_with_full_weight(self) -> None:
        cell = PeriodicCell(lattice_vectors=8.0 * np.eye(3))
        k_points, weights = monkhorst_pack_grid(cell, (1, 1, 1))
        np.testing.assert_allclose(k_points, [[0.0, 0.0, 0.0]])
        np.testing.assert_allclose(weights, [1.0])

    def test_2x2x2_has_eight_equally_weighted_points(self) -> None:
        cell = PeriodicCell(lattice_vectors=8.0 * np.eye(3))
        k_points, weights = monkhorst_pack_grid(cell, (2, 2, 2))
        self.assertEqual(k_points.shape, (8, 3))
        np.testing.assert_allclose(weights, np.full(8, 0.125))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)


class GradientOperatorTests(unittest.TestCase):
    def test_matches_analytic_plane_wave_derivative(self) -> None:
        box = 10.0
        cell = PeriodicCell(lattice_vectors=box * np.eye(3))
        grid = build_periodic_grid(
            cell, PeriodicGridSettings(spacing=0.25, expansion_order=8)
        )
        gx, gy, gz = build_gradient(grid)
        k = 2.0 * np.pi / box
        x, y, z = grid.coordinates.T
        f = np.cos(k * x)
        np.testing.assert_allclose(gx @ f, -k * np.sin(k * x), atol=1e-8)
        np.testing.assert_allclose(gy @ f, 0.0, atol=1e-12)
        np.testing.assert_allclose(gz @ f, 0.0, atol=1e-12)

    def test_consistent_with_existing_negative_laplacian(self) -> None:
        """-nabla^2 f should equal k^2*f for this same plane wave, using
        the already-validated build_negative_laplacian as ground truth."""
        box = 10.0
        cell = PeriodicCell(lattice_vectors=box * np.eye(3))
        grid = build_periodic_grid(
            cell, PeriodicGridSettings(spacing=0.25, expansion_order=8)
        )
        laplacian = build_negative_laplacian(grid)
        k = 2.0 * np.pi / box
        x = grid.coordinates[:, 0]
        f = np.cos(k * x)
        np.testing.assert_allclose(laplacian @ f, k * k * f, atol=1e-8)


class KPointHamiltonianFreeElectronTests(unittest.TestCase):
    """With V_eff=0 and no nonlocal projectors, H_k must reproduce the exact
    free-electron dispersion |k+G|**2 for a reciprocal lattice vector G --
    an analytic result independent of anything this module implements."""

    def test_empty_lattice_dispersion(self) -> None:
        import scipy.sparse as sp

        from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
        from parsec_python.V_ion.ionic_potential import NonlocalProjectorOperator

        box = 8.0
        cell = PeriodicCell(lattice_vectors=box * np.eye(3))
        grid = build_periodic_grid(
            cell, PeriodicGridSettings(spacing=0.25, expansion_order=8)
        )
        laplacian = build_negative_laplacian(grid)
        gradient = build_gradient(grid)
        empty_nonlocal = NonlocalProjectorOperator(
            sp.csc_matrix((grid.size, 0), dtype=complex), np.zeros(0), ()
        )
        k_point = np.array([0.3, -0.2, 0.15])
        hamiltonian = KPointKohnShamHamiltonian(
            laplacian, gradient, np.zeros(grid.size), empty_nonlocal, k_point
        )
        g_rec = (2.0 * np.pi / box) * np.array([1.0, -1.0, 2.0])
        x, y, z = grid.coordinates.T
        u = np.exp(1j * (g_rec[0] * x + g_rec[1] * y + g_rec[2] * z))
        expected_eigenvalue = float(np.dot(k_point + g_rec, k_point + g_rec))
        result = hamiltonian.apply(u)
        np.testing.assert_allclose(
            result, expected_eigenvalue * u, rtol=1e-5, atol=1e-5
        )


class KPointSCFReductionTests(unittest.TestCase):
    """The decisive integration check: sampling only k=(0,0,0) must converge
    to (within ordinary SCF tolerance of) the same result as the existing,
    independently-validated Gamma-only periodic path."""

    @classmethod
    def setUpClass(cls) -> None:
        box = 8.0
        cell = PeriodicCell(lattice_vectors=box * np.eye(3))
        problem = SinglePointInput(
            atoms=[Atom("Au", [box / 2, box / 2, box / 2])],
            pseudopotentials={"Au": SpeciesPotential(_AU_POTRE, 0)},
            grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
            periodic_cell=cell,
            scf=SCFSettings(
                max_iterations=80, number_of_states=10, convergence_criterion=2.0e-3
            ),
            hartree=HartreeSettings(multipole_order=4),
            eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
        )
        cls.system = prepare_periodic_single_point(problem)
        cls.gamma_reference = run_scf_gamma(cls.system)
        k_points, weights = monkhorst_pack_grid(cell, (1, 1, 1))
        cls.kpoint_result = run_scf_kpoints(cls.system, k_points, weights)

    def test_both_converge(self) -> None:
        self.assertTrue(self.gamma_reference.converged)
        self.assertTrue(self.kpoint_result.converged)

    def test_energies_agree_within_scf_tolerance(self) -> None:
        self.assertLess(
            abs(self.gamma_reference.energies.total - self.kpoint_result.energies.total),
            5.0e-3,
        )

    def test_rejects_unequal_weights(self) -> None:
        with self.assertRaises(ValueError):
            run_scf_kpoints(
                self.system,
                np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
                np.array([0.7, 0.3]),
            )


class MultiKPointSmokeTest(unittest.TestCase):
    def test_2x2x2_grid_converges(self) -> None:
        box = 8.0
        cell = PeriodicCell(lattice_vectors=box * np.eye(3))
        problem = SinglePointInput(
            atoms=[Atom("Au", [box / 2, box / 2, box / 2])],
            pseudopotentials={"Au": SpeciesPotential(_AU_POTRE, 0)},
            grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
            periodic_cell=cell,
            scf=SCFSettings(
                max_iterations=60, number_of_states=10, convergence_criterion=5.0e-3
            ),
            hartree=HartreeSettings(multipole_order=4),
            eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
        )
        system = prepare_periodic_single_point(problem)
        k_points, weights = monkhorst_pack_grid(cell, (2, 2, 2))
        result = run_scf_kpoints(system, k_points, weights)
        self.assertTrue(result.converged)
        self.assertTrue(np.isfinite(result.energies.total))


if __name__ == "__main__":
    unittest.main()
