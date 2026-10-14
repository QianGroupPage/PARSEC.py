"""Bloch-gauge invariance of the periodic k-point Hamiltonians.

``H_k`` acts on the periodic factor ``u_k`` of ``psi = e^{ik.r} u_k``.  For any
reciprocal-lattice vector ``G``, ``H_{k+G} = U H_k U^dagger`` with the diagonal
unitary ``U = diag(e^{-iG.r})``, so the two operators must have the same
spectrum (up to the finite-difference error of the kinetic stencil, which vanishes as
the grid is refined).

A nonlocal projector carrying only the constant per-image phase ``e^{ik.T}``
(the Bloch phase of the *psi* picture) is not covariant under ``U`` and fails
this test by several Rydberg, even though it is Hermitian and
time-reversal-symmetric -- so the operator-level Hermiticity tests alone could
not catch it.  The correct u-picture projector is
``e^{-ik.r} sum_T e^{ik.T} beta(r - R - T)``.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from parsec_python import Atom, PeriodicCell, PeriodicGridSettings, SpeciesPotential
from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from parsec_python.Hamiltonian.kpoint_spinor_operator import KPointSpinorKohnShamHamiltonian
from parsec_python.Laplacian import build_gradient, build_negative_laplacian
from parsec_python.V_ion import (
    NonlocalProjectorOperator,
    build_nonlocal_projectors,
    ewald_local_ionic_potential,
    load_pseudopotentials,
)

_AU_POTRE = (
    Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
)
_BOX = 8.0
# Calibrated on this cell (Au, expansion order 6): the finite-difference part
# of |E_k - E_{k+G}| falls with the grid (0.59, 0.17, 0.051, 0.015 Ry at
# h = 0.70, 0.50, 0.40, 0.32), whereas the old constant-phase projector
# plateaus at ~0.56 Ry independent of h.  At h = 0.32 this tolerance sits ~3x
# above the correct operator and ~14x below the buggy one.
_TOLERANCE = 0.04


class BlochGaugeInvarianceTests(unittest.TestCase):
    def setUp(self) -> None:
        cell = PeriodicCell(lattice_vectors=_BOX * np.eye(3))
        self.grid = build_periodic_grid(
            cell, PeriodicGridSettings(spacing=0.32, expansion_order=6)
        )
        self.atoms = [Atom("Au", [_BOX / 2, _BOX / 2, _BOX / 2 + 0.3])]
        self.specs = {"Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=True)}
        self.pots = load_pseudopotentials(self.specs, xc_functional="ca")
        self.lattice = cell.lattice_vectors
        self.veff = ewald_local_ionic_potential(
            self.grid, self.atoms, self.pots, self.specs, self.lattice
        )
        self.laplacian = build_negative_laplacian(self.grid)
        self.gradient = build_gradient(self.grid)
        self.k = np.array([0.3, -0.15, 0.22])
        self.shift = np.array([2.0 * np.pi / _BOX, 0.0, 0.0])

    def _scalar(self, k):
        nonlocal_k = build_nonlocal_projectors(
            self.grid, self.atoms, self.pots, self.specs, self.lattice, k_point=k
        )
        return KPointKohnShamHamiltonian(
            self.laplacian, self.gradient, self.veff, nonlocal_k, k
        )

    def _spinor(self, k):
        soc = tuple(
            build_spin_orbit_projectors(
                self.grid, self.atoms, self.pots, self.specs, self.lattice, k_point=k
            )
        )
        return KPointSpinorKohnShamHamiltonian(self._scalar(k), soc)

    @staticmethod
    def _lowest(hamiltonian, count):
        values = eigsh(
            hamiltonian.as_linear_operator(), k=count, which="SA", return_eigenvectors=False
        )
        return np.sort(values)

    def test_scalar_spectrum_is_invariant_under_a_reciprocal_shift(self) -> None:
        a = self._lowest(self._scalar(self.k), 8)
        b = self._lowest(self._scalar(self.k + self.shift), 8)
        np.testing.assert_allclose(a, b, atol=_TOLERANCE)

    def test_spinor_spectrum_is_invariant_under_a_reciprocal_shift(self) -> None:
        a = self._lowest(self._spinor(self.k), 8)
        b = self._lowest(self._spinor(self.k + self.shift), 8)
        np.testing.assert_allclose(a, b, atol=_TOLERANCE)

    def test_constant_phase_projector_violates_the_invariance(self) -> None:
        """Sensitivity check: undoing the ``e^{-ik.r}`` factor reconstructs the
        old (psi-picture) projector, which must fail the same test by a wide
        margin -- so the passing tests above are not vacuous."""

        def old_style(k):
            correct = build_nonlocal_projectors(
                self.grid, self.atoms, self.pots, self.specs, self.lattice, k_point=k
            )
            restored = sp.diags(np.exp(1j * (self.grid.coordinates @ k))) @ correct.projectors
            operator = NonlocalProjectorOperator(
                projectors=restored.tocsc(), signs=correct.signs, labels=correct.labels
            )
            return KPointKohnShamHamiltonian(
                self.laplacian, self.gradient, self.veff, operator, k
            )

        a = self._lowest(old_style(self.k), 8)
        b = self._lowest(old_style(self.k + self.shift), 8)
        self.assertGreater(np.max(np.abs(a - b)), 10 * _TOLERANCE)


if __name__ == "__main__":
    unittest.main()
