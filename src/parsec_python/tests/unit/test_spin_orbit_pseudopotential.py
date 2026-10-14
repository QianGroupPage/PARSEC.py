"""Source-parity tests for reading and combining relativistic POTRE channels.

``Au_POTRE.DAT``/``H_POTRE.DAT`` are copied verbatim from Fortran PARSEC's own
``examples/benchmarks/0d_AuH`` case -- the same AuH molecule used in Naveh,
Kronik, Tiago, and Chelikowsky, Phys. Rev. B 76, 153407 (2007), the paper that
introduced PARSEC's real-space spin-orbit formalism.  Au's pseudopotential is
generated with ``irel='rel'`` and carries explicit p/d spin-orbit channels;
H's does not.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from parsec_python import read_parsec_pseudopotential

_EXAMPLES = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH"
_AU_POTRE = _EXAMPLES / "Au_POTRE.DAT"
_H_POTRE = _EXAMPLES / "H_POTRE.DAT"


class SpinOrbitPseudopotentialParsingTests(unittest.TestCase):
    def test_relativistic_file_reports_spin_orbit_channels(self) -> None:
        potential = read_parsec_pseudopotential(_AU_POTRE)
        self.assertEqual(potential.relativity, "rel")
        self.assertEqual(potential.number_of_spin_orbit_channels, 2)
        self.assertTrue(potential.has_spin_orbit_channels)
        self.assertEqual(sorted(potential.spin_orbit_channel_potentials), [1, 2])
        # Ordinary s/p/d channels are still read exactly as before.
        self.assertEqual(sorted(potential.channel_potentials), [0, 1, 2])

    def test_non_relativistic_file_has_no_spin_orbit_channels(self) -> None:
        potential = read_parsec_pseudopotential(_H_POTRE)
        self.assertEqual(potential.number_of_spin_orbit_channels, 0)
        self.assertFalse(potential.has_spin_orbit_channels)
        self.assertEqual(potential.spin_orbit_channel_potentials, {})


class SpinOrbitProjectorConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.potential = read_parsec_pseudopotential(_AU_POTRE)
        # examples/benchmarks/0d_AuH/parsec.in selects Local_Component: s.
        self.local_l = 0

    def test_projectors_are_finite_for_both_so_channels(self) -> None:
        for angular_momentum in (1, 2):
            v_ion, v_so, sign = self.potential.spin_orbit_projectors(
                angular_momentum, self.local_l
            )
            self.assertEqual(v_ion.shape, self.potential.radii.shape)
            self.assertEqual(v_so.shape, self.potential.radii.shape)
            self.assertTrue(np.all(np.isfinite(v_ion)))
            self.assertTrue(np.all(np.isfinite(v_so)))
            self.assertIn(sign, (-1.0, 1.0))

    def test_recombination_inverts_back_to_independently_normalized_j_channels(
        self,
    ) -> None:
        """Round-trip ``pseudo.f90``'s j-split/normalize/recombine pipeline.

        Inverting ``v_ion = [(l+1)*proj_+ + l*proj_-]/(2l+1)`` and
        ``v_so = 2*(proj_+ - proj_-)/(2l+1)`` gives back
        ``proj_+ = v_ion + (l/2)*v_so`` and ``proj_- = v_ion - ((l+1)/2)*v_so``
        -- the same algebraic shape as the original ``V_l +/- ...*V_so_raw``
        split, now applied to the *normalized* projector pair.  This checks
        that identity against an independent re-implementation of the
        per-j-channel Kleinman-Bylander normalization, catching a sign or
        weight regression in either the split or the recombination step that
        the shape/finiteness checks above would not.
        """
        from parsec_python.Pseudopotential.radial_quadrature import (
            parsec_radial_integral,
        )

        for angular_momentum in (1, 2):
            channel_v = self.potential.channel_potentials[angular_momentum]
            so_raw = self.potential.spin_orbit_channel_potentials[angular_momentum]
            local_v = self.potential.channel_potentials[self.local_l]
            radial_wave = self.potential.radial_wavefunctions[angular_momentum]
            radii = self.potential.radii

            def _independent_normalize(v_j: np.ndarray) -> np.ndarray:
                delta_v = v_j - local_v
                denominator = parsec_radial_integral(
                    radii, radial_wave * radial_wave * delta_v
                )
                projector = delta_v * radial_wave / radii
                projector /= np.sqrt(abs(denominator))
                if projector.size > 16:
                    projector[:8] = projector[16]
                return projector

            v_plus = channel_v + 0.5 * angular_momentum * so_raw
            v_minus = channel_v - 0.5 * (angular_momentum + 1) * so_raw
            expected_proj_plus = _independent_normalize(v_plus)
            expected_proj_minus = _independent_normalize(v_minus)

            v_ion, v_so, _ = self.potential.spin_orbit_projectors(
                angular_momentum, self.local_l
            )
            recovered_proj_plus = v_ion + 0.5 * angular_momentum * v_so
            recovered_proj_minus = v_ion - 0.5 * (angular_momentum + 1) * v_so

            np.testing.assert_allclose(
                recovered_proj_plus, expected_proj_plus, rtol=1e-10, atol=1e-14
            )
            np.testing.assert_allclose(
                recovered_proj_minus, expected_proj_minus, rtol=1e-10, atol=1e-14
            )

    def test_rejects_unsupported_angular_momentum(self) -> None:
        with self.assertRaises(ValueError):
            self.potential.spin_orbit_projectors(0, self.local_l)


if __name__ == "__main__":
    unittest.main()
