"""Molecular dynamics with a net force (FDET keeps it): the centre of mass is integrated, not held.

PARSEC 1.5 removes the centre-of-mass velocity in every microcanonical step (``moldyn.f90:461-503``) also
under FDET, whose forces keep their net part (``forcnloc.F90:869-875``): the motion that net force drives is
then thrown away and the energy is not conserved (H2 tip: 0.07 Ry lost in 4 steps of dt 10).  With
``keep_center_of_mass`` the integrator keeps it.
"""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python.MolecularDynamics.moldyn import moldyn_step, start_molecular_dynamics
from parsec_python.MolecularDynamics.settings import MolecularDynamicsSettings


class NetForceDynamicsTest(unittest.TestCase):
    """Two atoms in a constant external force F (Ry/bohr): E = -sum F.x, a = F/(2m) in Hartree units."""

    def run_steps(self, keep: bool, steps: int = 6):
        settings = MolecularDynamicsSettings(
            enabled=True, time_step=10.0, step_num=steps, initial_temperature=300.0, final_temperature=300.0,
            rng_seed=7, use_test_rng=True,
        )
        masses = np.array([1837.0, 1837.0])
        force = np.array([[-1.0, 0.0, 0.2], [-0.8, 0.0, 0.0]])
        positions = np.array([[0.1, 0.0, 1.9], [0.1, 0.0, 3.3]])
        state = start_molecular_dynamics(settings, positions, masses)
        random = settings.random_source()
        conserved, centres = [], []
        for _ in range(steps):
            energy = -float(np.sum(force * state.current_positions))
            step = moldyn_step(settings, state, force, energy, random=random, keep_center_of_mass=keep)
            conserved.append(step.conserved_energy)
            centres.append(state.current_positions.mean(axis=0))
            state = step.state
        return np.array(conserved), np.array(centres), masses, force, settings

    def test_the_net_force_moves_the_centre_of_mass(self):
        conserved, centres, masses, force, settings = self.run_steps(keep=True)
        # Beeman's scheme is exact for a constant acceleration: the energy is conserved to round-off, and the centre
        # moves by a t^2/2 with a = F_net/(2 M) (the Maxwell start has no centre-of-mass velocity).
        np.testing.assert_allclose(conserved - conserved[0], 0.0, atol=1.0e-9)
        acceleration = force.sum(axis=0) / (2.0 * masses.sum())
        times = settings.time_step * np.arange(len(centres))
        np.testing.assert_allclose(
            centres - centres[0], 0.5 * acceleration[None, :] * times[:, None] ** 2, rtol=1.0e-9, atol=1.0e-12
        )

    def test_holding_the_centre_of_mass_loses_energy(self):
        # The PARSEC 1.5 behaviour (keep_center_of_mass false): the work of the net force is not counted.
        conserved, centres, *_ = self.run_steps(keep=False)
        self.assertGreater(abs(conserved[-1] - conserved[0]), 1.0e-3)


if __name__ == "__main__":
    unittest.main()
