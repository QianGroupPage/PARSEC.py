"""Correctness tests for PARSEC's perturbative spin-orbit diagonalization.

Uses the real ``Au_POTRE.DAT`` from Fortran PARSEC's own ``0d_AuH`` example
(the Naveh et al., PRB 76, 153407 (2007) validation system) so the SOC
projector data (V_ion/V_so, KB signs) is genuine, not synthetic.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from parsec_python import Atom, GridSettings, SpeciesPotential
from parsec_python.Grid import build_cluster_grid
from parsec_python.V_ion import load_pseudopotentials
from parsec_python.models import SpinPolarizedSinglePointResult, EnergyBreakdown
from parsec_python.Eigensolvers.perturbative_soc import (
    _lsxy_block,
    _lzsz_block,
    _projector_overlaps,
    build_spin_orbit_projectors,
    perturbative_spin_orbit_correction,
)

_AU_POTRE = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"


def _grid_and_projectors():
    grid = build_cluster_grid(GridSettings(spacing=0.6, radius=3.0, expansion_order=4))
    atoms = [Atom("Au", [0.0, 0.0, 0.0])]
    specifications = {"Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=True)}
    potentials = load_pseudopotentials(specifications, xc_functional="ca")
    projectors = build_spin_orbit_projectors(grid, atoms, potentials, specifications)
    return grid, atoms, potentials, specifications, projectors


def _random_normalized_wavefunctions(grid_size: int, n_states: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wavefunctions = rng.normal(size=(grid_size, n_states))
    return wavefunctions / np.linalg.norm(wavefunctions, axis=0, keepdims=True)


class SpinOrbitMatrixHermiticityTests(unittest.TestCase):
    """The strongest available correctness check: lzsz/lsxy transcribed from
    pls.F90 must give an exactly Hermitian 2N x 2N matrix.  In particular the
    up-down and down-up blocks are computed by two independent calls (with
    swapped input/output channels and opposite ladder sign) that have no
    code in common beyond the shared helper -- if the lm-index bookkeeping or
    a sign in the transcription were wrong, they would not match beyond
    accidental cases, let alone to floating-point roundoff.
    """

    @classmethod
    def setUpClass(cls) -> None:
        grid, cls.atoms, cls.potentials, cls.specifications, cls.projectors = (
            _grid_and_projectors()
        )
        cls.grid = grid
        cls.n_states = 5
        wf_up = _random_normalized_wavefunctions(grid.size, cls.n_states, seed=0)
        wf_down = _random_normalized_wavefunctions(grid.size, cls.n_states, seed=1)
        cls.dotio_up, cls.dotso_up = _projector_overlaps(cls.projectors, wf_up)
        cls.dotio_down, cls.dotso_down = _projector_overlaps(cls.projectors, wf_down)

    def test_finds_the_soc_atom(self) -> None:
        self.assertEqual(len(self.projectors), 1)
        self.assertEqual(self.projectors[0].v_ion.shape[1], 8)
        self.assertEqual(self.projectors[0].sign, (1.0, -1.0))

    def test_diagonal_blocks_are_exactly_hermitian(self) -> None:
        up_up = _lzsz_block(self.dotio_up, self.dotso_up, spin_sign=1)
        down_down = _lzsz_block(self.dotio_down, self.dotso_down, spin_sign=-1)
        np.testing.assert_allclose(up_up, up_up.conj().T, atol=1e-12)
        np.testing.assert_allclose(down_down, down_down.conj().T, atol=1e-12)

    def test_off_diagonal_blocks_are_mutual_hermitian_conjugates(self) -> None:
        up_down = _lsxy_block(
            self.dotio_down, self.dotso_down, self.dotio_up, self.dotso_up, spin_sign=-1
        )
        down_up = _lsxy_block(
            self.dotio_up, self.dotso_up, self.dotio_down, self.dotso_down, spin_sign=1
        )
        np.testing.assert_allclose(up_down, down_up.conj().T, atol=1e-12)

    def test_off_diagonal_blocks_are_not_trivially_zero(self) -> None:
        """A degenerate all-zero result would make the Hermiticity check
        vacuous; confirm the SOC coupling is actually nonzero here."""
        up_down = _lsxy_block(
            self.dotio_down, self.dotso_down, self.dotio_up, self.dotso_up, spin_sign=-1
        )
        self.assertGreater(np.max(np.abs(up_down)), 1e-8)


class PerturbativeSpinOrbitCorrectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid, self.atoms, self.potentials, self.specifications, _ = (
            _grid_and_projectors()
        )
        self.n_states = 4
        wf_up = _random_normalized_wavefunctions(self.grid.size, self.n_states, seed=2)
        wf_down = _random_normalized_wavefunctions(self.grid.size, self.n_states, seed=3)
        eig_up = np.array([-2.0, -1.5, -1.0, -0.5])
        eig_down = np.array([-1.9, -1.4, -0.9, -0.4])
        self.scf_result = SpinPolarizedSinglePointResult(
            converged=True,
            iterations=1,
            atoms=tuple(self.atoms),
            electron_count=4.0,
            eigenvalues_up=eig_up,
            eigenvalues_down=eig_down,
            occupations_up=np.array([1.0, 1.0, 0.0, 0.0]),
            occupations_down=np.array([1.0, 1.0, 0.0, 0.0]),
            wavefunctions_up=wf_up,
            wavefunctions_down=wf_down,
            fermi_level=-1.2,
            density_up=np.zeros(self.grid.size),
            density_down=np.zeros(self.grid.size),
            energies=EnergyBreakdown(
                eigenvalue=0.0,
                hartree=0.0,
                integral_vxc_rho=0.0,
                exchange_correlation=0.0,
                electron_ion=0.0,
                ion_ion=0.0,
                electronic=0.0,
                total=0.0,
            ),
        )

    def test_returns_2n_real_sorted_eigenvalues(self) -> None:
        result = perturbative_spin_orbit_correction(
            self.grid, self.scf_result, self.atoms, self.potentials, self.specifications
        )
        self.assertEqual(result.eigenvalues.shape, (2 * self.n_states,))
        self.assertTrue(np.all(np.isfinite(result.eigenvalues)))
        self.assertTrue(np.all(np.diff(result.eigenvalues) >= -1e-12))

    def test_spinor_coefficients_are_unitary_columns(self) -> None:
        result = perturbative_spin_orbit_correction(
            self.grid, self.scf_result, self.atoms, self.potentials, self.specifications
        )
        norms = np.sum(np.abs(result.spinor_coefficients_up) ** 2, axis=0) + np.sum(
            np.abs(result.spinor_coefficients_down) ** 2, axis=0
        )
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_magnetic_moment_matches_spinor_weights(self) -> None:
        result = perturbative_spin_orbit_correction(
            self.grid, self.scf_result, self.atoms, self.potentials, self.specifications
        )
        expected = np.sum(np.abs(result.spinor_coefficients_up) ** 2, axis=0) - np.sum(
            np.abs(result.spinor_coefficients_down) ** 2, axis=0
        )
        np.testing.assert_allclose(result.magnetic_moment, expected, atol=1e-12)

    def test_rejects_mismatched_channel_state_counts(self) -> None:
        mismatched = SpinPolarizedSinglePointResult(
            **{
                **vars(self.scf_result),
                "eigenvalues_down": self.scf_result.eigenvalues_down[:-1],
            }
        )
        with self.assertRaises(ValueError):
            perturbative_spin_orbit_correction(
                self.grid, mismatched, self.atoms, self.potentials, self.specifications
            )


if __name__ == "__main__":
    unittest.main()
