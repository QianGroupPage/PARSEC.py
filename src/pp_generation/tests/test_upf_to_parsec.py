"""Resampling of ``src/tools/upf_to_parsec.py`` onto the PARSEC radial mesh."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[2] / "tools" / "upf_to_parsec.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("upf_to_parsec_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


class ResamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        # The mesh of an FHI98PP table: 2.47 % between neighbouring radii.
        self.radius = tuple(7.8125e-4 * math.exp(0.0244 * index) for index in range(473))
        self.log_radius = tuple(math.log(value) for value in self.radius)

    @staticmethod
    def potential(radius: float) -> float:
        """A screened well of the depth and width of an oxygen potential (Ry)."""

        return -30.0 * math.exp(-(radius / 0.7) ** 2) - 12.0 * math.erf(radius) / radius

    def errors(self, interpolate) -> float:
        values = tuple(self.potential(value) for value in self.radius)
        worst = 0.0
        for index in range(len(self.radius) - 1):
            target = math.sqrt(self.radius[index] * self.radius[index + 1])
            worst = max(
                worst,
                abs(
                    interpolate(self.radius, self.log_radius, values, target)
                    - self.potential(target)
                ),
            )
        return worst

    def test_cubic_resampling_is_accurate_between_the_mesh_points(self) -> None:
        linear = self.errors(tool.log_linear_interpolate)
        cubic = self.errors(tool.log_cubic_interpolate)
        self.assertGreater(linear, 1.0e-3)
        self.assertLess(cubic, 1.0e-5)

    def test_tabulated_radius_returns_its_value(self) -> None:
        values = tuple(self.potential(value) for value in self.radius)
        for index in (0, 1, 200, len(self.radius) - 2, len(self.radius) - 1):
            self.assertEqual(
                tool.log_cubic_interpolate(
                    self.radius, self.log_radius, values, self.radius[index]
                ),
                values[index],
            )

    def test_ends_and_short_meshes(self) -> None:
        values = tuple(self.potential(value) for value in self.radius)
        beyond = 2.0 * self.radius[-1]
        self.assertEqual(
            tool.log_cubic_interpolate(self.radius, self.log_radius, values, beyond),
            values[-1],
        )
        with self.assertRaises(tool.ConversionError):
            tool.log_cubic_interpolate(
                self.radius, self.log_radius, values, 0.5 * self.radius[0]
            )
        # Fewer than four points: the linear value.
        target = math.sqrt(self.radius[0] * self.radius[1])
        self.assertEqual(
            tool.log_cubic_interpolate(
                self.radius[:3], self.log_radius[:3], values[:3], target
            ),
            tool.log_linear_interpolate(
                self.radius[:3], self.log_radius[:3], values[:3], target
            ),
        )

    def test_reals_without_an_exponent_letter(self) -> None:
        self.assertEqual(tool.parse_float("1.5D-03"), 1.5e-3)
        self.assertEqual(tool.parse_float("0.123456789012-103"), 0.123456789012e-103)
        self.assertEqual(tool.parse_float("-4.2+101"), -4.2e101)
        with self.assertRaises(ValueError):
            tool.parse_float("1.0E-0x")
        with self.assertRaises(ValueError):
            tool.parse_float("abc")

    def test_both_interpolations_are_offered(self) -> None:
        self.assertEqual(set(tool.INTERPOLATIONS), {"cubic", "linear"})
        arguments = tool.build_parser().parse_args(["in.UPF", "out_POTRE.DAT"])
        self.assertEqual(arguments.interpolation, "cubic")


if __name__ == "__main__":
    unittest.main()
