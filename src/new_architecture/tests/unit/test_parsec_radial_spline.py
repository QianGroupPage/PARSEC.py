"""Source-parity tests for PARSEC's pseudopotential radial spline."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from new_architecture import (
    Atom,
    GridSettings,
    ParsecRadialSpline,
    SpeciesPotential,
    build_nonlocal_projectors,
    parsec_radial_integral,
    read_parsec_pseudopotential,
)


class ParsecRadialSplineTests(unittest.TestCase):
    def test_matches_compiled_fortran_spline_and_splint(self) -> None:
        """Compare against coefficients and values produced by spline.f90."""
        radii = np.array([0.1, 0.3, 0.8, 1.5])
        values = np.array([1.8, 1.1, 0.2, 0.0])
        spline = ParsecRadialSpline.from_positive_grid(
            radii,
            values,
            padding_width=2,
            origin_value=2.0,
        )

        expected_knots = np.array(
            [-3.0, -2.0, -1.0, 0.0, 0.1, 0.3, 0.8, 1.5]
        )
        expected_second_derivatives = np.array(
            [
                0.20489397429848166,
                -0.4097879485969633,
                1.4342578200893716,
                -5.327243331760523,
                -17.143224902162157,
                9.093296372366705,
                1.7960601182380869,
                0.326459736799324,
            ]
        )
        query = np.array(
            [0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.55, 0.8, 1.0, 1.5]
        )
        expected_values = np.array(
            [
                2.0,
                1.9596099084244638,
                1.9140440426462015,
                1.8,
                1.4701248213244889,
                1.1000000000000001,
                0.47985380483430029,
                0.2,
                0.084545573690354897,
                0.0,
            ]
        )

        np.testing.assert_array_equal(spline.knots, expected_knots)
        np.testing.assert_allclose(
            spline.second_derivatives,
            expected_second_derivatives,
            rtol=0.0,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            spline(query),
            expected_values,
            rtol=0.0,
            atol=2.0e-14,
        )

    def test_padding_width_is_expansion_order_half_width(self) -> None:
        spline = ParsecRadialSpline.from_positive_grid(
            np.array([0.1, 0.4]),
            np.array([3.0, 1.0]),
            padding_width=4,
        )
        np.testing.assert_array_equal(
            spline.knots[:6],
            [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0],
        )
        np.testing.assert_array_equal(spline.values[:6], np.full(6, 3.0))

    def test_local_and_core_paths_use_the_same_padded_spline(self) -> None:
        path = Path(__file__).parents[1] / "data" / "H_CORE_POTRE.DAT"
        potential = read_parsec_pseudopotential(path)
        query = np.array(
            [0.0, 0.5 * potential.radii[0], potential.radii[0], 0.2]
        )

        local_reference = ParsecRadialSpline.from_positive_grid(
            potential.radii,
            potential.channel_potentials[0],
            padding_width=4,
        )
        core_reference = ParsecRadialSpline.from_positive_grid(
            potential.radii,
            potential.core_density,
            padding_width=4,
        )
        np.testing.assert_allclose(
            potential.local_potential(
                query,
                0,
                use_spline=True,
                spline_padding_width=4,
            ),
            local_reference(query),
        )
        np.testing.assert_allclose(
            potential.interpolate_density(
                query,
                core=True,
                use_spline=True,
                spline_padding_width=4,
            ),
            core_reference(query),
        )

    def test_parsec_radial_quadrature_endpoint_weights(self) -> None:
        radii = np.array([0.1, 0.4, 1.0])
        values = np.array([2.0, 3.0, 5.0])
        expected = (
            0.5 * 0.4 * 2.0
            + 0.5 * (1.0 - 0.1) * 3.0
            + (1.0 - 0.4) * 5.0
        )
        self.assertAlmostEqual(
            parsec_radial_integral(radii, values),
            expected,
        )

    def test_nonlocal_spline_clamps_only_its_radial_query(self) -> None:
        path = (
            Path(__file__).parents[2]
            / "benchmarks"
            / "0d_benzene"
            / "C_POTRE.DAT"
        )
        potential = read_parsec_pseudopotential(path)
        specification = SpeciesPotential(
            path,
            local_angular_momentum=1,
            use_spline=True,
        )

        # A single diagnostic point inside the first positive POTRE radius is
        # enough to distinguish PARSEC's r -> max(r, r_first) rule.  The
        # builder only needs this small public subset of RealSpaceGrid here.
        query_radius = 0.5 * potential.radii[0]
        grid = SimpleNamespace(
            coordinates=np.array([[query_radius, 0.0, 0.0]]),
            size=1,
            volume_element=1.0,
            settings=GridSettings(
                spacing=1.0,
                radius=1.0,
                expansion_order=8,
            ),
        )
        operator = build_nonlocal_projectors(
            grid,
            [Atom("C", [0.0, 0.0, 0.0])],
            {"C": potential},
            {"C": specification},
        )

        radial_projector, _ = potential.radial_projector(0, 1)
        spline = ParsecRadialSpline.from_positive_grid(
            potential.radii,
            radial_projector,
            padding_width=4,
        )
        y00 = 0.28209479177387814
        actual = float(operator.projectors[0, 0])
        clamped = y00 * spline(potential.radii[0])
        unclamped = y00 * spline(query_radius)
        self.assertAlmostEqual(actual, clamped, places=14)
        self.assertGreater(abs(actual - unclamped), 5.0e-15)


if __name__ == "__main__":
    unittest.main()
