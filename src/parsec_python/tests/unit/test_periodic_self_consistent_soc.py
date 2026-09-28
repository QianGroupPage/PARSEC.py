"""Tests for periodic self-consistent spin-orbit coupling (k-point sampling
combined with SOC).

Uses the real ``Au_POTRE.DAT`` from Fortran PARSEC's own ``0d_AuH`` example.
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
from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
from parsec_python.Grid import monkhorst_pack_grid
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from parsec_python.Hamiltonian.kpoint_spinor_operator import KPointSpinorKohnShamHamiltonian
from parsec_python.Laplacian import build_gradient, build_negative_laplacian
from parsec_python.SCF.kpoints_soc import run_self_consistent_soc_kpoints
from parsec_python.SCF.pbc import prepare_periodic_single_point
from parsec_python.V_ion import build_nonlocal_projectors, load_pseudopotentials

_AU_POTRE = (
    Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
)


def _build_hamiltonian(grid, atoms, pots, specs, lattice_vectors, k_point, veff):
    nonlocal_k = build_nonlocal_projectors(
        grid, atoms, pots, specs, lattice_vectors, k_point=k_point
    )
    soc_k = tuple(
        build_spin_orbit_projectors(
            grid, atoms, pots, specs, lattice_vectors, k_point=k_point
        )
    )
    negative_laplacian = build_negative_laplacian(grid)
    gradient = build_gradient(grid)
    scalar_h = KPointKohnShamHamiltonian(
        negative_laplacian, gradient, veff, nonlocal_k, k_point
    )
    return KPointSpinorKohnShamHamiltonian(scalar_h, soc_k)


class PeriodicSpinOrbitProjectorHermiticityTests(unittest.TestCase):
    """Decisive operator-level check before trusting any SCF result built
    on top: the full periodic spinor Hamiltonian (kinetic + ordinary
    nonlocal + spin-orbit, all Bloch-phase weighted) must be Hermitian at a
    genuinely nonzero k-point."""

    def test_hermitian_at_nonzero_k(self) -> None:
        box = 10.0
        cell = PeriodicCell(lattice_vectors=box * np.eye(3))
        grid = build_periodic_grid(
            cell, PeriodicGridSettings(spacing=0.7, expansion_order=6)
        )
        atoms = [Atom("Au", [box / 2, box / 2, box / 2])]
        specs = {"Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=True)}
        pots = load_pseudopotentials(specs, xc_functional="ca")
        veff = np.zeros(grid.size)

        k = np.array([0.25, -0.1, 0.18])
        hamiltonian = _build_hamiltonian(
            grid, atoms, pots, specs, cell.lattice_vectors, k, veff
        )
        rng = np.random.default_rng(11)
        n = 4
        u = rng.normal(size=(2 * grid.size, n)) + 1j * rng.normal(size=(2 * grid.size, n))
        u /= np.linalg.norm(u, axis=0, keepdims=True)
        v = rng.normal(size=(2 * grid.size, n)) + 1j * rng.normal(size=(2 * grid.size, n))
        v /= np.linalg.norm(v, axis=0, keepdims=True)

        lhs = np.vdot(u, hamiltonian.apply(v))
        rhs = np.vdot(hamiltonian.apply(u), v)
        self.assertLess(abs(lhs - rhs), 1e-10)


class TimeReversalSymmetryTests(unittest.TestCase):
    """E_n(k) = E_n(-k) is a hard analytic requirement for any real
    (non-magnetic) spin-orbit Hamiltonian -- an independent check of the
    Bloch-phase bookkeeping in both the ordinary and spin-orbit k-point
    projectors together."""

    def test_eigenvalues_match_at_k_and_minus_k(self) -> None:
        from parsec_python.Eigensolvers import (
            ChebFFSettings,
            ChebDavSettings,
            EigvalSettings,
            SubspaceSettings,
            solve_eigval,
        )

        box = 10.0
        cell = PeriodicCell(lattice_vectors=box * np.eye(3))
        grid = build_periodic_grid(
            cell, PeriodicGridSettings(spacing=0.7, expansion_order=6)
        )
        atoms = [Atom("Au", [box / 2, box / 2, box / 2])]
        specs = {"Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=True)}
        pots = load_pseudopotentials(specs, xc_functional="ca")
        from parsec_python.V_ion import ewald_local_ionic_potential

        veff = ewald_local_ionic_potential(
            grid, atoms, pots, specs, cell.lattice_vectors
        )

        settings = EigvalSettings(
            safety_buffer=6,
            initial_method="chebff",
            chebff=ChebFFSettings(
                polynomial_degree=20, filter_cycles=8, lanczos_steps=8, block_size=6
            ),
            chebdav=ChebDavSettings(),
            subspace=SubspaceSettings(
                polynomial_degree=15, degree_delta=3, lanczos_steps=8, block_size=6
            ),
        )

        def diagonalize(k_point: np.ndarray) -> np.ndarray:
            hamiltonian = _build_hamiltonian(
                grid, atoms, pots, specs, cell.lattice_vectors, k_point, veff
            )
            solution = solve_eigval(
                hamiltonian.as_linear_operator(), 10, settings=settings, state=None
            )
            return np.sort(solution.eigenvalues)

        k = np.array([0.3, -0.15, 0.22])
        eig_k = diagonalize(k)
        eig_minus_k = diagonalize(-k)
        np.testing.assert_allclose(eig_k, eig_minus_k, atol=1e-8)

    def test_doubly_degenerate_at_generic_k_for_centrosymmetric_atom(self) -> None:
        """A single atom at the cell center is centrosymmetric; combined
        with time reversal (PT symmetry), bands must be doubly degenerate
        at *every* k, not only at TRIM points."""
        from parsec_python.Eigensolvers import (
            ChebFFSettings,
            ChebDavSettings,
            EigvalSettings,
            SubspaceSettings,
            solve_eigval,
        )
        from parsec_python.V_ion import ewald_local_ionic_potential

        box = 10.0
        cell = PeriodicCell(lattice_vectors=box * np.eye(3))
        grid = build_periodic_grid(
            cell, PeriodicGridSettings(spacing=0.7, expansion_order=6)
        )
        atoms = [Atom("Au", [box / 2, box / 2, box / 2])]
        specs = {"Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=True)}
        pots = load_pseudopotentials(specs, xc_functional="ca")
        veff = ewald_local_ionic_potential(
            grid, atoms, pots, specs, cell.lattice_vectors
        )
        hamiltonian = _build_hamiltonian(
            grid,
            atoms,
            pots,
            specs,
            cell.lattice_vectors,
            np.array([0.3, -0.15, 0.22]),
            veff,
        )
        settings = EigvalSettings(
            safety_buffer=6,
            initial_method="chebff",
            chebff=ChebFFSettings(
                polynomial_degree=20, filter_cycles=8, lanczos_steps=8, block_size=6
            ),
            chebdav=ChebDavSettings(),
            subspace=SubspaceSettings(
                polynomial_degree=15, degree_delta=3, lanczos_steps=8, block_size=6
            ),
        )
        solution = solve_eigval(
            hamiltonian.as_linear_operator(), 10, settings=settings, state=None
        )
        eigenvalues = np.sort(solution.eigenvalues)
        for i in range(0, len(eigenvalues) - 1, 2):
            self.assertAlmostEqual(eigenvalues[i], eigenvalues[i + 1], places=6)


class PeriodicSelfConsistentSOCSCFTests(unittest.TestCase):
    """Integration-level check: at the Gamma point (a TRIM point), the
    converged self-consistent result must show the same Kramers-pair
    signature as the cluster path."""

    @classmethod
    def setUpClass(cls) -> None:
        box = 8.0
        cell = PeriodicCell(lattice_vectors=box * np.eye(3))
        problem = SinglePointInput(
            atoms=[Atom("Au", [box / 2, box / 2, box / 2])],
            pseudopotentials={"Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=True)},
            grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
            periodic_cell=cell,
            scf=SCFSettings(
                max_iterations=60, number_of_states=20, convergence_criterion=5.0e-3
            ),
            hartree=HartreeSettings(multipole_order=4),
            eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
        )
        system = prepare_periodic_single_point(problem)
        k_points, weights = monkhorst_pack_grid(cell, (1, 1, 1))
        cls.result = run_self_consistent_soc_kpoints(system, k_points, weights)

    def test_converges(self) -> None:
        self.assertTrue(self.result.converged)

    def test_kramers_pairs_at_gamma(self) -> None:
        eigenvalues = self.result.eigenvalues
        moments = self.result.magnetic_moment_per_state
        for i in range(0, len(eigenvalues) - 3, 2):
            self.assertAlmostEqual(eigenvalues[i], eigenvalues[i + 1], delta=1.0e-4)
            self.assertAlmostEqual(moments[i] + moments[i + 1], 0.0, delta=0.05)

    def test_rejects_unequal_weights(self) -> None:
        box = 8.0
        cell = PeriodicCell(lattice_vectors=box * np.eye(3))
        problem = SinglePointInput(
            atoms=[Atom("Au", [box / 2, box / 2, box / 2])],
            pseudopotentials={"Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=True)},
            grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
            periodic_cell=cell,
            scf=SCFSettings(max_iterations=1, number_of_states=10),
            hartree=HartreeSettings(multipole_order=4),
            eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
        )
        system = prepare_periodic_single_point(problem)
        with self.assertRaises(ValueError):
            run_self_consistent_soc_kpoints(
                system,
                np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
                np.array([0.7, 0.3]),
            )


class MultiKPointSelfConsistentSOCSmokeTest(unittest.TestCase):
    def test_2x2x2_grid_converges(self) -> None:
        box = 8.0
        cell = PeriodicCell(lattice_vectors=box * np.eye(3))
        problem = SinglePointInput(
            atoms=[Atom("Au", [box / 2, box / 2, box / 2])],
            pseudopotentials={"Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=True)},
            grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
            periodic_cell=cell,
            scf=SCFSettings(
                max_iterations=60, number_of_states=20, convergence_criterion=1.0e-2
            ),
            hartree=HartreeSettings(multipole_order=4),
            eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
        )
        system = prepare_periodic_single_point(problem)
        k_points, weights = monkhorst_pack_grid(cell, (2, 2, 2))
        result = run_self_consistent_soc_kpoints(system, k_points, weights)
        self.assertTrue(result.converged)
        self.assertTrue(np.isfinite(result.energies.total))
        self.assertEqual(result.eigenvalues.size, 8 * 20)


if __name__ == "__main__":
    unittest.main()
