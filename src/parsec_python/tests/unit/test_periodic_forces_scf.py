"""Forces of a converged periodic calculation against the derivative of its SCF total energy.

H2 in a small periodic box.  The forces are exact derivatives of the discrete energy functional for fixed orbitals, so
the Hellmann-Feynman total equals the derivative of the SCF energy of the same grid, egg-box effects included (here the
first atom sits on a grid point and the coarse grid makes the egg-box force about as large as the bond force).  The SCF
energy is not smooth on that scale, so its derivative is Richardson-extrapolated from steps of 0.005 and 0.01 bohr.

The forces differentiate the reference local potential of ``ewald_local_ionic_potential``, so these tests run with the FFT
provider of the acceleration package switched off, and with the cubic spline for the periodic local potential
(``ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE``, the default), the derivative of which is exact.

The five SCF runs take about two minutes (the setup of the reference periodic path, not the SCF, with the 118-bohr
hydrogen table), so the test runs when ``PARSEC_SLOW_TESTS`` is set; each force term has its own fast finite-difference
test (``test_ewald_forces``, ``test_periodic_local_forces``, ``test_periodic_nonlocal_forces``).
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from parsec_python.Forces import compute_periodic_forces
from parsec_python.V_ion import ewald
from parsec_python.models import (
    Atom,
    EigensolverSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.SCF import prepare_periodic_single_point, run_scf
from parsec_python.SCF.kpoints import run_scf_kpoints

_H_POTRE = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "H_POTRE.DAT"


def _problem(shift: float) -> SinglePointInput:
    return SinglePointInput(
        atoms=[Atom("H", [3.0 + shift, 3.0, 3.0]), Atom("H", [4.3, 3.0, 3.0])],
        pseudopotentials={"H": SpeciesPotential(_H_POTRE, 0, use_spline=True)},
        grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
        periodic_cell=PeriodicCell(lattice_vectors=6.0 * np.eye(3)),
        scf=SCFSettings(max_iterations=60, number_of_states=3, convergence_criterion=1.0e-9),
        hartree=HartreeSettings(relative_tolerance=1e-10, absolute_tolerance=1e-13, max_iterations=4000),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-9),
    )


def _scf(shift: float):
    system = prepare_periodic_single_point(_problem(shift))
    return system, run_scf(system)


@unittest.skipUnless(os.environ.get("PARSEC_SLOW_TESTS"), "set PARSEC_SLOW_TESTS=1 (about two minutes)")
class PeriodicForceSCFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.addCleanup(setattr, ewald, "_local_potential_provider", ewald._local_potential_provider)
        ewald.set_local_potential_provider(None)
        # exact derivative of the spline (the default for the periodic local potential)
        self.addCleanup(setattr, ewald, "LOCAL_POTENTIAL_USES_CUBIC_SPLINE", ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE)
        ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE = True

    def test_total_force_is_the_derivative_of_the_scf_energy(self) -> None:
        system, result = _scf(0.0)
        self.assertTrue(result.converged)
        forces = compute_periodic_forces(system, result)
        raw = forces.unsymmetrized[0, 0]
        self.assertGreater(abs(raw), 0.02)

        def central(step: float) -> float:
            plus, minus = _scf(+step)[1], _scf(-step)[1]
            self.assertTrue(plus.converged and minus.converged)
            return -(plus.energies.total - minus.energies.total) / (2.0 * step)

        numeric = (4.0 * central(0.005) - central(0.01)) / 3.0
        self.assertAlmostEqual(float(raw), numeric, delta=5.0e-4)  # of a force of 0.03: the residual of the extrapolation

        # the net force is removed from the reported total, and a crystal has no torque
        np.testing.assert_allclose(forces.total.sum(axis=0), 0.0, atol=1.0e-12)
        np.testing.assert_array_equal(forces.torque, 0.0)
        # the terms add up to the unsymmetrized force
        np.testing.assert_allclose(
            forces.ion_ion + forces.local.total + forces.nonlocal_pseudopotential, forces.unsymmetrized
        )


_K_POINTS = np.array([[0.21, -0.37, 0.11], [-0.40, 0.15, 0.30]])


def _scf_kpoints(shift: float, k_points, weights, forces: bool = False):
    system = prepare_periodic_single_point(_problem(shift))
    return system, run_scf_kpoints(system, np.asarray(k_points, dtype=float), np.asarray(weights, dtype=float), compute_forces=forces)


@unittest.skipUnless(os.environ.get("PARSEC_SLOW_TESTS"), "set PARSEC_SLOW_TESTS=1 (about three minutes)")
class PeriodicKPointForceSCFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.addCleanup(setattr, ewald, "_local_potential_provider", ewald._local_potential_provider)
        ewald.set_local_potential_provider(None)
        # exact derivative of the spline (the default for the periodic local potential)
        self.addCleanup(setattr, ewald, "LOCAL_POTENTIAL_USES_CUBIC_SPLINE", ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE)
        ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE = True

    def test_gamma_through_the_kpoint_driver_and_time_reversal(self) -> None:
        system = prepare_periodic_single_point(_problem(0.0))
        gamma = compute_periodic_forces(system, run_scf(system))
        through = run_scf_kpoints(system, np.zeros((1, 3)), np.ones(1), compute_forces=True)
        np.testing.assert_allclose(through.nonlocal_forces, gamma.nonlocal_pseudopotential, atol=1.0e-7)
        np.testing.assert_allclose(compute_periodic_forces(system, through).total, gamma.total, atol=1.0e-7)
        # a k-point and its time-reversed partner carry conjugate states: {k, -k} gives the forces of k alone
        single = run_scf_kpoints(system, _K_POINTS[:1], np.ones(1), compute_forces=True)
        pair = run_scf_kpoints(system, np.stack([_K_POINTS[0], -_K_POINTS[0]]), np.full(2, 0.5), compute_forces=True)
        np.testing.assert_allclose(pair.nonlocal_forces, single.nonlocal_forces, atol=1.0e-7)
        np.testing.assert_allclose(
            compute_periodic_forces(system, pair).total, compute_periodic_forces(system, single).total, atol=1.0e-7
        )

    def test_a_kpoint_result_without_its_nonlocal_forces_is_refused(self) -> None:
        system, result = _scf_kpoints(0.0, _K_POINTS, [0.5, 0.5])
        self.assertIsNone(result.nonlocal_forces)
        with self.assertRaises(ValueError):
            compute_periodic_forces(system, result)

    def test_kpoint_force_is_the_derivative_of_the_scf_energy(self) -> None:
        weights = [0.5, 0.5]
        system, result = _scf_kpoints(0.0, _K_POINTS, weights, forces=True)
        self.assertTrue(result.converged)
        raw = compute_periodic_forces(system, result).unsymmetrized[0, 0]

        def central(step: float) -> float:
            plus, minus = _scf_kpoints(+step, _K_POINTS, weights)[1], _scf_kpoints(-step, _K_POINTS, weights)[1]
            return -(plus.energies.total - minus.energies.total) / (2.0 * step)

        numeric = (4.0 * central(0.0025) - central(0.005)) / 3.0
        self.assertAlmostEqual(float(raw), numeric, delta=5.0e-5)


if __name__ == "__main__":
    unittest.main()
