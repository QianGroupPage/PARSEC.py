"""CHEBDAV on complex Hermitian operators (Bloch Hamiltonians at k != 0).

The complex arithmetic (conjugate-transposed projections, Hermitian mirroring
of the upper triangle) must leave the real path untouched, reduce to it
exactly for a complex operator that happens to be real, and agree with an
exact eigensolver for a genuinely complex Bloch Hamiltonian.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

from parsec_python import Atom, PeriodicCell, PeriodicGridSettings, SpeciesPotential
from parsec_python.Eigensolvers.chebdav import ChebDavSettings, run_chebdav
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from parsec_python.Hamiltonian.operator import KohnShamHamiltonian
from parsec_python.Laplacian import build_gradient, build_negative_laplacian
from parsec_python.V_ion import build_nonlocal_projectors, load_pseudopotentials

_AU_POTRE = (
    Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
)
_WANTED = 26


class ComplexChebDavTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        box = 9.0
        cls.cell = PeriodicCell(lattice_vectors=box * np.eye(3))
        cls.grid = build_periodic_grid(
            cls.cell, PeriodicGridSettings(spacing=0.5, expansion_order=6)
        )
        cls.laplacian = build_negative_laplacian(cls.grid)
        cls.gradient = build_gradient(cls.grid)
        cls.atoms = [Atom("Au", [3.5, 4.5, 4.5]), Atom("Au", [6.3, 4.0, 5.0])]
        cls.specs = {"Au": SpeciesPotential(_AU_POTRE, 0)}
        cls.pots = load_pseudopotentials(cls.specs, xc_functional="ca")
        cls.potential = 0.3 * np.cos(2 * np.pi * cls.grid.coordinates[:, 0] / box)
        cls.settings = ChebDavSettings(polynomial_degree=20, convergence_tolerance=1e-4)

    def _bloch_operator(self, k_point):
        projectors = build_nonlocal_projectors(
            self.grid,
            self.atoms,
            self.pots,
            self.specs,
            self.cell.lattice_vectors,
            k_point=k_point,
        )
        return KPointKohnShamHamiltonian(
            self.laplacian, self.gradient, self.potential, projectors, k_point
        ).as_linear_operator()

    def test_complex_operator_at_gamma_reproduces_the_real_path(self) -> None:
        real_projectors = build_nonlocal_projectors(
            self.grid, self.atoms, self.pots, self.specs, self.cell.lattice_vectors
        )
        real = run_chebdav(
            KohnShamHamiltonian(
                self.laplacian, self.potential, real_projectors
            ).as_linear_operator(),
            _WANTED,
            settings=self.settings,
        )
        cplx = run_chebdav(self._bloch_operator(np.zeros(3)), _WANTED, settings=self.settings)
        self.assertEqual(real.vectors.dtype, np.float64)
        self.assertEqual(cplx.vectors.dtype, np.complex128)
        np.testing.assert_allclose(cplx.eigenvalues, real.eigenvalues, atol=1e-9)
        self.assertEqual(
            cplx.state.truly_converged_states, real.state.truly_converged_states
        )
        np.testing.assert_allclose(cplx.residual_norms, real.residual_norms, atol=1e-9)

    def test_matches_exact_eigenvalues_at_a_nonzero_k_point(self) -> None:
        operator = self._bloch_operator(np.array([0.21, -0.13, 0.08]))
        exact = np.sort(eigsh(operator, k=_WANTED, which="SA", tol=1e-10)[0])
        result = run_chebdav(operator, _WANTED, settings=self.settings)
        self.assertEqual(result.vectors.dtype, np.complex128)
        # The lowest states are converged; the last few are the working buffer.
        np.testing.assert_allclose(result.eigenvalues[:20], exact[:20], atol=1e-4)
        vectors = result.vectors
        np.testing.assert_allclose(
            vectors.conj().T @ vectors, np.eye(_WANTED), atol=1e-10
        )
        residual = np.asarray(operator @ vectors) - vectors * result.eigenvalues
        self.assertLess(np.linalg.norm(residual, axis=0)[:20].max(), 1e-2)
        self.assertTrue(np.all(np.isreal(result.eigenvalues)))


if __name__ == "__main__":
    unittest.main()
