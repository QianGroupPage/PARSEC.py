"""Source-parity tests for the spin-polarized CA/PZ (LSDA) functional.

Formulas are transcribed from Fortran PARSEC's ``exc_spn.f90``, ``icorr=='ca'``
branch.
"""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python import ca_lda, ca_lda_spin_polarized


class UnpolarizedLimitTests(unittest.TestCase):
    def test_equal_spin_densities_reduce_to_unpolarized_ca_lda(self) -> None:
        """At zeta=0, LSDA must exactly reproduce the unpolarized functional."""
        rho = np.array([0.0005, 0.01, 0.05, 0.3, 1.0, 3.0, 10.0])
        volume_element = 0.37

        unpolarized = ca_lda(rho, volume_element)
        polarized = ca_lda_spin_polarized(rho / 2.0, rho / 2.0, volume_element)

        np.testing.assert_allclose(
            polarized.potential_up, unpolarized.potential, rtol=1e-12
        )
        np.testing.assert_allclose(
            polarized.potential_down, unpolarized.potential, rtol=1e-12
        )
        np.testing.assert_allclose(
            polarized.energy_density, unpolarized.energy_density, rtol=1e-10
        )
        self.assertAlmostEqual(
            polarized.total_energy, unpolarized.total_energy, places=10
        )

    def test_with_core_density_still_reduces_to_unpolarized(self) -> None:
        rho = np.array([0.02, 0.5, 4.0])
        core = np.array([0.01, 0.01, 0.01])
        volume_element = 0.2

        unpolarized = ca_lda(rho, volume_element, core_density=core)
        polarized = ca_lda_spin_polarized(
            rho / 2.0, rho / 2.0, volume_element, core_density=core
        )
        np.testing.assert_allclose(
            polarized.potential_up, unpolarized.potential, rtol=1e-12
        )


class SpinPolarizedBehaviorTests(unittest.TestCase):
    def test_majority_spin_channel_has_more_attractive_potential(self) -> None:
        """Exchange favors same-spin density depletion: the majority channel
        should sit at a more negative (more attractive) XC potential."""
        result = ca_lda_spin_polarized(
            np.array([5.0, 0.01]), np.array([1.0, 0.001]), volume_element=0.37
        )
        self.assertTrue(np.all(np.isfinite(result.potential_up)))
        self.assertTrue(np.all(np.isfinite(result.potential_down)))
        self.assertTrue(np.all(result.potential_up < result.potential_down))

    def test_zero_minority_density_point_contributes_nothing(self) -> None:
        """Matches exc_spn.f90's ``if (minval(rh) > zero)`` guard: a point
        with one spin density exactly zero gets zero XC potential/energy."""
        result = ca_lda_spin_polarized(
            np.array([1.0]), np.array([0.0]), volume_element=0.5
        )
        self.assertEqual(result.potential_up[0], 0.0)
        self.assertEqual(result.potential_down[0], 0.0)
        self.assertEqual(result.energy_density[0], 0.0)

    def test_rejects_mismatched_shapes(self) -> None:
        with self.assertRaises(ValueError):
            ca_lda_spin_polarized(np.array([1.0, 2.0]), np.array([1.0]), 0.5)


if __name__ == "__main__":
    unittest.main()
