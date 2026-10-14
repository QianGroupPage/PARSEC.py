"""Physics-validation tests for self-consistent spin-orbit coupling.

Uses the real ``Au_POTRE.DAT`` from Fortran PARSEC's own ``0d_AuH`` example.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from parsec_python import (
    Atom,
    EigensolverSettings,
    GridSettings,
    HartreeSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
from parsec_python.SCF.self_consistent_soc import run_self_consistent_soc
from parsec_python.SCF.single_point import prepare_single_point

_AU_POTRE = (
    Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
)


def _au_problem(**scf_overrides) -> SinglePointInput:
    return SinglePointInput(
        atoms=[Atom("Au", [0.0, 0.0, 0.0])],
        pseudopotentials={"Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=True)},
        grid=GridSettings(spacing=0.6, radius=6.0, expansion_order=6),
        scf=SCFSettings(
            max_iterations=40,
            number_of_states=20,
            convergence_criterion=5.0e-3,
            **scf_overrides,
        ),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
    )


class SelfConsistentSOCKramersDegeneracyTests(unittest.TestCase):
    """A spin-orbit-coupled Hamiltonian with no external magnetic field has
    exact time-reversal symmetry, so every eigenvalue is at least doubly
    (Kramers) degenerate, with the two partners carrying opposite <S_z>.
    This is not built into the code anywhere -- it is an emergent physical
    consequence of a correct SOC Hamiltonian, making it a strong check."""

    @classmethod
    def setUpClass(cls) -> None:
        system = prepare_single_point(_au_problem())
        soc_projectors = build_spin_orbit_projectors(
            system.grid,
            system.atoms,
            system.pseudopotentials,
            _au_problem().pseudopotentials,
        )
        cls.result = run_self_consistent_soc(system, tuple(soc_projectors))

    def test_converges(self) -> None:
        self.assertTrue(self.result.converged)

    def test_eigenvalues_form_kramers_pairs(self) -> None:
        """Excludes the topmost pair: with number_of_states requested exactly
        at the working-subspace size, the highest-index (virtual, physically
        irrelevant) states sit right at the Chebyshev filter window's edge,
        where CHEBFF's fixed-cycle policy (no per-state residual test, see
        chebff.py's module docstring) is documented to be least accurate."""
        eigenvalues = self.result.eigenvalues
        for i in range(0, len(eigenvalues) - 3, 2):
            self.assertAlmostEqual(eigenvalues[i], eigenvalues[i + 1], delta=1.0e-4)

    def test_kramers_partners_have_opposite_moment(self) -> None:
        moments = self.result.magnetic_moment_per_state
        for i in range(0, len(moments) - 3, 2):
            self.assertAlmostEqual(moments[i] + moments[i + 1], 0.0, delta=0.05)

    def test_moments_are_physically_bounded(self) -> None:
        moments = self.result.magnetic_moment_per_state
        self.assertTrue(np.all(moments >= -1.0 - 1e-9))
        self.assertTrue(np.all(moments <= 1.0 + 1e-9))

    def test_spinor_columns_are_normalized(self) -> None:
        norms = np.sum(np.abs(self.result.spinors) ** 2, axis=0)
        np.testing.assert_allclose(norms, 1.0, atol=1e-8)

    def test_energy_close_to_the_no_soc_spin_polarized_reference(self) -> None:
        """SOC is a comparatively small correction for AuH-scale systems, so
        the total energy should stay close to the (independently validated)
        spin-polarized-without-SOC result on a similar grid, not diverge."""
        self.assertLess(abs(self.result.energies.total - (-67.04)), 1.0)


class SelfConsistentSOCEigensolverGuardTests(unittest.TestCase):
    def test_rejects_chebdav(self) -> None:
        problem = _au_problem()
        problem = SinglePointInput(
            atoms=problem.atoms,
            pseudopotentials=problem.pseudopotentials,
            grid=problem.grid,
            scf=problem.scf,
            hartree=problem.hartree,
            eigensolver=EigensolverSettings(method="chebdav", tolerance=1.0e-6),
        )
        system = prepare_single_point(problem)
        soc_projectors = build_spin_orbit_projectors(
            system.grid, system.atoms, system.pseudopotentials, problem.pseudopotentials
        )
        with self.assertRaises(NotImplementedError):
            run_self_consistent_soc(system, tuple(soc_projectors))


if __name__ == "__main__":
    unittest.main()
