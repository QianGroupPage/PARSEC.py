"""Forces of a converged periodic calculation against the derivative of its SCF total energy.

H2 in a small periodic box.  The forces are exact derivatives of the discrete energy functional for fixed orbitals, so
the Hellmann-Feynman total equals the derivative of the SCF energy of the same grid, egg-box effects included (here the
first atom sits on a grid point and the coarse grid makes the egg-box force about as large as the bond force).  The SCF
energy is not smooth on that scale, so its derivative is Richardson-extrapolated from steps of 0.005 and 0.01 bohr.

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


if __name__ == "__main__":
    unittest.main()
