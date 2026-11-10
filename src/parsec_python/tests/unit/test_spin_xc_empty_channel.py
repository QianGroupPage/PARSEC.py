"""Collinear spin XC with an empty channel (a decision of this port, 2026-11-05).

PARSEC's ``exc_spn.f90`` evaluates the spin functional only where both spin densities are positive
(``minval(rh) > zero``), and ``V_xc.ca_lda.ca_lda_spin_polarized`` did the same: a channel that the Fermi
occupations leave exactly empty (zero beyond 36.7 kT, ``flevel.f90``) gave no exchange-correlation at all, in
either channel, and a one-electron spin cell flipped its moment for dozens of SCF iterations.  The spin PBE of
``V_xc.pbe.pbe_spin_polarized`` floors the channels at 1e-16, where the derivative of the exact phi(zeta)
diverges, and gave the empty channel potentials of 1e7 Ry.  Both now take the fully polarized limit
(zeta = +-1) wherever the total density is positive; points with both channels positive keep their bits.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from parsec_python.Grid.cluster import build_cluster_grid
from parsec_python.models import GridSettings
from parsec_python.V_xc.ca_lda import ca_lda, ca_lda_spin_polarized
from parsec_python.V_xc.ca_lda_spin import ca_lda_spin
from parsec_python.V_xc.pbe import pbe_spin_polarized, pbe_spin_polarized_partials
from parsec_python.V_xc.pbe_spin import pbe_spin

_DENSITIES = np.array([2.0, 0.3, 0.1, 0.01, 1.0e-3, 1.0e-6, 1.0e-12])


def _relative(value, reference) -> float:
    value, reference = np.asarray(value), np.asarray(reference)
    return float(np.max(np.abs(value - reference)) / np.max(np.abs(reference)))


class CASpinEmptyChannelTests(unittest.TestCase):
    def evaluate(self, up, down):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no 0**(-1/3) or division warnings at zeta = +-1
            return ca_lda_spin_polarized(up, down, 1.0)

    def test_full_polarization_is_the_limit_of_a_vanishing_minority(self) -> None:
        empty = np.zeros_like(_DENSITIES)
        for label, (up, down, tiny_up, tiny_down) in {
            "zeta=+1": (_DENSITIES, empty, _DENSITIES, empty + 1.0e-300),
            "zeta=-1": (empty, _DENSITIES, empty + 1.0e-300, _DENSITIES),
        }.items():
            with self.subTest(label):
                result = self.evaluate(up, down)
                tiny = self.evaluate(tiny_up, tiny_down)
                np.testing.assert_array_equal(result.energy_density, tiny.energy_density)
                # the empty channel's exchange (2 * 1e-300)**(1/3) is ~1e-100: the potentials agree to that
                np.testing.assert_allclose(result.potential_up, tiny.potential_up, rtol=0, atol=1e-90)
                np.testing.assert_allclose(result.potential_down, tiny.potential_down, rtol=0, atol=1e-90)
                self.assertTrue(np.all(np.isfinite(result.potential_up)) and np.all(np.isfinite(result.potential_down)))
                self.assertTrue(np.all(result.energy_density < 0.0))
                # the other implementation of the same functional (V_xc.ca_lda_spin), written independently
                other = ca_lda_spin(up, down, 1.0)
                self.assertLess(_relative(result.energy_density, other.energy_density), 1e-14)
                self.assertLess(_relative(result.potential_up, other.potential_up), 1e-14)
                self.assertLess(_relative(result.potential_down, other.potential_down), 1e-14)

    def test_the_channels_are_mirror_images_and_zeta_zero_is_the_unpolarized_formula(self) -> None:
        empty = np.zeros_like(_DENSITIES)
        plus, minus = self.evaluate(_DENSITIES, empty), self.evaluate(empty, _DENSITIES)
        np.testing.assert_array_equal(plus.potential_up, minus.potential_down)
        np.testing.assert_array_equal(plus.potential_down, minus.potential_up)
        self.assertEqual(plus.total_energy, minus.total_energy)
        # zeta = 0: the unpolarized CA-LDA (the other limit of the interpolation)
        half = self.evaluate(0.5 * _DENSITIES, 0.5 * _DENSITIES)
        unpolarized = ca_lda(_DENSITIES, 1.0)
        self.assertLess(_relative(half.potential_up, unpolarized.potential), 1e-14)
        self.assertLess(_relative(half.energy_density, unpolarized.energy_density), 1e-14)
        # full polarization lowers the exchange: 2**(1/3) of the unpolarized one at the same density
        self.assertTrue(np.all(plus.energy_density < half.energy_density))

    def test_points_with_both_channels_keep_their_bits(self) -> None:
        up = np.array([0.4, 0.2, 0.0, 0.05, 0.3, -1.0e-15])
        down = np.array([0.1, 0.0, 0.0, 0.05, -1.0e-15, 0.2])
        mixed = self.evaluate(up, down)
        both = np.array([0, 3])
        alone = self.evaluate(up[both], down[both])
        for field in ("potential_up", "potential_down", "energy_density"):
            self.assertEqual(getattr(mixed, field)[both].tobytes(), getattr(alone, field).tobytes())
        # both channels empty: nothing; a channel below zero is taken as empty
        self.assertEqual((mixed.potential_up[2], mixed.potential_down[2], mixed.energy_density[2]), (0.0, 0.0, 0.0))
        clipped = self.evaluate(np.array([0.3, 0.0]), np.array([0.0, 0.2]))
        np.testing.assert_array_equal(mixed.energy_density[[4, 5]], clipped.energy_density)


class PBESpinEmptyChannelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = build_cluster_grid(GridSettings(spacing=0.45, radius=5.0, expansion_order=12))
        x = cls.grid.coordinates
        cls.density = 1.5 * np.exp(-np.sum((x - [0.0, 0.0, 0.3]) ** 2, axis=1)) + 0.4 * np.exp(
            -np.sum((x - [1.4, 0.0, -0.8]) ** 2, axis=1)
        )

    def test_full_polarization_is_finite_and_equals_the_regularized_form(self) -> None:
        empty = np.zeros_like(self.density)
        for label, up, down in (("zeta=+1", self.density, empty), ("zeta=-1", empty, self.density)):
            with self.subTest(label):
                result = pbe_spin_polarized(up, down, self.grid)
                reference = pbe_spin(up, down, self.grid)
                # before: the empty channel's potential reached 9e7 Ry here
                self.assertLess(float(np.max(np.abs(result.potential_down))), 10.0)
                self.assertLess(float(np.max(np.abs(result.potential_up))), 10.0)
                self.assertLess(_relative(result.potential_up, reference.potential_up), 1e-13)
                self.assertLess(_relative(result.potential_down, reference.potential_down), 1e-13)
                self.assertAlmostEqual(result.total_energy, reference.total_energy, delta=1e-12)

    def test_points_with_both_channels_keep_their_bits(self) -> None:
        up = np.array([0.4, 0.2, 0.05, 0.3])
        down = np.array([0.1, 0.0, 0.05, 0.0])
        s_uu, s_ud, s_dd = np.array([0.02, 0.01, 0.003, 0.05]), np.array([0.004, 0.0, 0.001, 0.0]), np.array([0.01, 0.0, 0.002, 0.0])
        mixed = pbe_spin_polarized_partials(up, down, s_uu, s_ud, s_dd)
        both = np.array([0, 2])
        alone = pbe_spin_polarized_partials(up[both], down[both], s_uu[both], s_ud[both], s_dd[both])
        for value, reference in zip(mixed, alone):
            self.assertEqual(value[both].tobytes(), reference.tobytes())
        self.assertTrue(all(np.all(np.isfinite(value)) for value in mixed))


if __name__ == "__main__":
    unittest.main()
