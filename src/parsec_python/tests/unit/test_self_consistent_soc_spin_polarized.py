"""Physics-validation tests for self-consistent spin-orbit coupling combined
with collinear spin polarization.

Uses the real ``Au_POTRE.DAT``/``H_POTRE.DAT`` from Fortran PARSEC's own
``0d_AuH`` example.
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
from parsec_python.Hamiltonian.spinor_operator import SpinorKohnShamHamiltonian
from parsec_python.SCF.self_consistent_soc_spin_polarized import (
    run_self_consistent_soc_spin_polarized,
    spinor_spin_densities,
)
from parsec_python.SCF.single_point import prepare_single_point

_AU_POTRE = (
    Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
)


def _au_problem(**scf_overrides) -> SinglePointInput:
    scf_kwargs = dict(
        max_iterations=40,
        number_of_states=20,
        convergence_criterion=5.0e-3,
        spin_polarized=True,
        self_consistent_spin_orbit=True,
    )
    scf_kwargs.update(scf_overrides)
    return SinglePointInput(
        atoms=[Atom("Au", [0.0, 0.0, 0.0])],
        pseudopotentials={"Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=True)},
        grid=GridSettings(spacing=0.6, radius=6.0, expansion_order=6),
        scf=SCFSettings(**scf_kwargs),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
    )


class SpinorHamiltonianXCDeltaHermiticityTests(unittest.TestCase):
    """The ``xc_delta`` Zeeman-like term is a real diagonal field with
    opposite sign on the two spinor channels -- trivially Hermitian on its
    own, but this checks the *combined* operator (kinetic + nonlocal + SOC +
    xc_delta) stays Hermitian, since a sign or indexing slip in how
    ``xc_delta`` is broadcast onto ``psi_up``/``psi_down`` would not
    necessarily break at any single term."""

    def test_hermitian_with_xc_delta_and_soc(self) -> None:
        system = prepare_single_point(_au_problem())
        soc_projectors = tuple(
            build_spin_orbit_projectors(
                system.grid,
                system.atoms,
                system.pseudopotentials,
                system.input.pseudopotentials,
            )
        )
        rng = np.random.default_rng(7)
        xc_delta = 0.1 * rng.normal(size=system.grid.size)
        scalar_hamiltonian = system.hamiltonian(system.ionic_potential)
        hamiltonian = SpinorKohnShamHamiltonian(
            scalar_hamiltonian.negative_laplacian,
            scalar_hamiltonian.effective_potential,
            scalar_hamiltonian.nonlocal_operator,
            soc_projectors,
            xc_delta=xc_delta,
        )
        n = 4
        size = hamiltonian.shape[0]
        u = rng.normal(size=(size, n)) + 1j * rng.normal(size=(size, n))
        v = rng.normal(size=(size, n)) + 1j * rng.normal(size=(size, n))
        lhs = np.vdot(u, hamiltonian.apply(v))
        rhs = np.vdot(hamiltonian.apply(u), v)
        self.assertLess(abs(lhs - rhs), 2e-10)

    def test_xc_delta_none_matches_omitting_the_field(self) -> None:
        system = prepare_single_point(_au_problem())
        scalar_hamiltonian = system.hamiltonian(system.ionic_potential)
        without_field = SpinorKohnShamHamiltonian(
            scalar_hamiltonian.negative_laplacian,
            scalar_hamiltonian.effective_potential,
            scalar_hamiltonian.nonlocal_operator,
            (),
        )
        with_none = SpinorKohnShamHamiltonian(
            scalar_hamiltonian.negative_laplacian,
            scalar_hamiltonian.effective_potential,
            scalar_hamiltonian.nonlocal_operator,
            (),
            xc_delta=None,
        )
        rng = np.random.default_rng(3)
        v = rng.normal(size=without_field.shape[0]) + 1j * rng.normal(
            size=without_field.shape[0]
        )
        np.testing.assert_array_equal(without_field.apply(v), with_none.apply(v))


class SpinorSpinDensitiesTests(unittest.TestCase):
    def test_sums_to_the_ordinary_spinor_density(self) -> None:
        from parsec_python.SCF.self_consistent_soc import spinor_density_from_orbitals

        rng = np.random.default_rng(11)
        n_grid, n_states = 5, 3
        spinors = rng.normal(size=(2 * n_grid, n_states)) + 1j * rng.normal(
            size=(2 * n_grid, n_states)
        )
        occupations = np.array([1.0, 1.0, 0.3])
        volume_element = 0.4
        density_up, density_down = spinor_spin_densities(
            spinors, occupations, volume_element
        )
        expected_total = spinor_density_from_orbitals(spinors, occupations, volume_element)
        np.testing.assert_allclose(density_up + density_down, expected_total)


class SelfConsistentSOCSpinPolarizedSCFTests(unittest.TestCase):
    """Isolated Au atom: known [Xe]4f14 5d10 6s1 ground state, unpaired 6s
    electron, so the converged net moment should be nonzero (unlike the
    zero-moment AuH molecule the Fortran cross-reference below uses)."""

    @classmethod
    def setUpClass(cls) -> None:
        system = prepare_single_point(_au_problem())
        soc_projectors = tuple(
            build_spin_orbit_projectors(
                system.grid,
                system.atoms,
                system.pseudopotentials,
                system.input.pseudopotentials,
            )
        )
        cls.result = run_self_consistent_soc_spin_polarized(system, soc_projectors)

    def test_converges(self) -> None:
        self.assertTrue(self.result.converged)

    def test_net_moment_is_nonzero(self) -> None:
        net_moment = float(
            np.dot(self.result.occupations, self.result.magnetic_moment_per_state)
        )
        self.assertGreater(abs(net_moment), 0.3)

    def test_spinor_columns_are_normalized(self) -> None:
        norms = np.sum(np.abs(self.result.spinors) ** 2, axis=0)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


class SelfConsistentSOCSpinPolarizedGuardTests(unittest.TestCase):
    def test_rejects_without_spin_polarized(self) -> None:
        system = prepare_single_point(_au_problem(spin_polarized=False))
        with self.assertRaises(ValueError):
            run_self_consistent_soc_spin_polarized(system, ())

    def test_rejects_without_self_consistent_spin_orbit(self) -> None:
        system = prepare_single_point(
            _au_problem(self_consistent_spin_orbit=False)
        )
        with self.assertRaises(ValueError):
            run_self_consistent_soc_spin_polarized(system, ())


if __name__ == "__main__":
    unittest.main()
