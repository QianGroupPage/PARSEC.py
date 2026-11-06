"""Fourier filtering of the pseudopotentials (``fourier_f.f90``).

The transforms are compared with the table Fortran PARSEC writes
(``fourier_Si.dat``, ``data/fourier_filter_reference.json``), the filter with
the identity it must approach for a wide filter, and the input with the
rules of ``usrinputfile.F90:1897-1907`` (periodic systems only).
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import spherical_jn

from parsec_python.Input import ParsecInputError, parse_parsec_input
from parsec_python.Pseudopotential import apply_fourier_filter, read_parsec_pseudopotential
from parsec_python.Pseudopotential.fourier_filter import (
    DELTA_Q,
    filter_weights,
    filtered_function,
    fourier_transform,
    reciprocal_mesh_size,
    spherical_bessel,
)
from parsec_python.Pseudopotential.radial_derivatives import valence_density_table
from parsec_python.V_ion import apply_fourier_filters
from parsec_python.acceleration.cli import _reference_only_features
from parsec_python.models import SpeciesPotential

DATA = Path(__file__).resolve().parents[1] / "data"


class BesselTests(unittest.TestCase):
    def test_against_scipy(self) -> None:
        x = np.linspace(0.5, 60.0, 400)
        for order in range(4):
            np.testing.assert_allclose(
                spherical_bessel(order, x), spherical_jn(order, x), rtol=1e-9, atol=1e-12
            )
        small = np.array([0.0, 1e-4, 9e-4])
        np.testing.assert_allclose(spherical_bessel(0, small), spherical_jn(0, small), atol=1e-14)
        np.testing.assert_allclose(spherical_bessel(1, small), spherical_jn(1, small), atol=1e-14)
        self.assertEqual(spherical_bessel(2, np.array([0.0]))[0], 0.0)


class TransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = json.loads((DATA / "fourier_filter_reference.json").read_text())
        cls.potential = read_parsec_pseudopotential(DATA / "Si_POTRE.DAT")

    def test_mesh_size_of_fortran(self) -> None:
        # parsec.out: "Si     nql=  820     delq= 0.015".
        self.assertEqual(
            reciprocal_mesh_size(self.reference["spacing"], 0.8, 0.8), self.reference["rows"]
        )
        self.assertEqual(reciprocal_mesh_size(0.5, 0.0, 0.0), 2 * round(np.pi / 0.5 / DELTA_Q) + 4)

    def test_transforms_against_fortran(self) -> None:
        potential = self.potential
        count = self.reference["rows"]
        index = np.asarray(self.reference["index"])
        fortran = np.asarray(self.reference["values"])
        transforms = {
            1: fourier_transform(
                potential,
                potential.channel_potentials[1],
                0,
                count,
                coulomb=2.0 * potential.ionic_charge,
            ),
            3: fourier_transform(
                potential, valence_density_table(potential, read_from_file=False), 0, count
            ),
            4: fourier_transform(potential, potential.radial_projector(0, 1)[0], 0, count),
            6: fourier_transform(potential, potential.radial_projector(2, 1)[0], 2, count),
        }
        transforms[1][0] = -4.0 * potential.ionic_charge / np.pi
        np.testing.assert_allclose(DELTA_Q * index, fortran[:, 0], rtol=5e-4)
        for column, values in transforms.items():
            expected = fortran[:, column]
            # g16.4 keeps four significant digits.
            np.testing.assert_allclose(
                values[index], expected, rtol=6e-4, atol=6e-4 * np.max(np.abs(expected))
            )

    def test_wide_filter_is_nearly_the_identity(self) -> None:
        potential = self.potential
        radii = potential.radii
        smooth = np.exp(-((radii / 1.2) ** 2))
        spacing = 0.4
        alpha = 3.0
        count = reciprocal_mesh_size(spacing, alpha, 0.0)
        transform = fourier_transform(potential, smooth, 0, count)
        back = filtered_function(radii, transform, 0, spacing, alpha, 0.0)
        inside = radii < 6.0
        np.testing.assert_allclose(back[inside], smooth[inside], atol=2e-6)
        # A p-like function through the l = 1 transform.
        shaped = radii * np.exp(-((radii / 1.0) ** 2))
        transform = fourier_transform(potential, shaped, 1, count)
        back = filtered_function(radii, transform, 1, spacing, alpha, 0.0)
        np.testing.assert_allclose(back[inside], shaped[inside], atol=2e-6)
        # A narrow filter smooths.
        narrow = filtered_function(radii, transform, 1, spacing, 0.3, 0.0)
        self.assertGreater(np.max(np.abs(narrow[inside] - shaped[inside])), 1e-3)

    def test_filter_weights(self) -> None:
        count = 400
        sharp = filter_weights(count, 0.5, 0.5, 0.0)
        cutoff = int(0.5 * np.pi / 0.5 / DELTA_Q) + 1
        self.assertTrue(np.all(sharp[cutoff:] == 0.0))
        smooth = filter_weights(count, 0.5, 0.5, 5.0)
        self.assertTrue(np.all(smooth[:cutoff] == sharp[:cutoff]))
        self.assertGreater(smooth[cutoff + 5], 0.0)
        largest = (np.sqrt(10.0 / 5.0) + 1.0) * (0.5 * np.pi / 0.5 / DELTA_Q)
        self.assertTrue(np.all(smooth[int(largest) + 2 :] == 0.0))


class FilteredPotentialTests(unittest.TestCase):
    def test_tables_replaced(self) -> None:
        potential = read_parsec_pseudopotential(DATA / "Si_POTRE.DAT")
        self.assertIs(apply_fourier_filter(potential, 1, spacing=0.5, alpha=0.0), potential)
        filtered = apply_fourier_filter(potential, 1, spacing=0.5, alpha=0.8, beta1=5.0)
        self.assertEqual(sorted(filtered.filtered_projectors), [0, 2])
        table, sign = filtered.radial_projector(0, 1)
        original, original_sign = potential.radial_projector(0, 1)
        self.assertEqual(sign, original_sign)
        self.assertFalse(np.array_equal(table, original))
        # The filtered projector stays close to the original away from the core.
        self.assertLess(np.max(np.abs(table - original)), 0.2 * np.max(np.abs(original)))
        np.testing.assert_array_equal(
            valence_density_table(filtered, read_from_file=False),
            filtered.filtered_valence_density,
        )
        # The local potential is not filtered (its periodic consumer reads the
        # transform of the unfiltered table).
        np.testing.assert_array_equal(filtered.channel_potentials[1], potential.channel_potentials[1])
        # No core correction in this file: Core_Filter changes nothing.
        np.testing.assert_array_equal(filtered.core_density, potential.core_density)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            apply_fourier_filter(potential, 1, spacing=0.5, alpha=-1.0)

    def test_projector_slope_is_dfint(self) -> None:
        # PARSEC 1.5 math/fourier_f.f90:267: dvw = dfint, the radial
        # derivative of the filtered projector (the previous tree stored fint).
        from parsec_python.Pseudopotential.fourier_filter import filtered_derivative
        from parsec_python.Pseudopotential.radial_derivatives import (
            radial_projector_with_derivative,
        )

        potential = read_parsec_pseudopotential(DATA / "Si_POTRE.DAT")
        spacing, alpha, beta1 = 0.5, 0.8, 5.0
        filtered = apply_fourier_filter(potential, 1, spacing=spacing, alpha=alpha, beta1=beta1)
        self.assertEqual(sorted(filtered.filtered_projector_slopes), [0, 2])
        self.assertEqual(potential.filtered_projector_slopes, {})
        radii = potential.radii
        count = reciprocal_mesh_size(spacing, alpha, alpha)
        for angular_momentum in (0, 2):
            table, slope, sign = radial_projector_with_derivative(filtered, angular_momentum, 1)
            np.testing.assert_array_equal(table, filtered.radial_projector(angular_momentum, 1)[0])
            self.assertEqual(sign, filtered.radial_projector(angular_momentum, 1)[1])
            np.testing.assert_array_equal(slope, filtered.filtered_projector_slopes[angular_momentum])
            # Central differences of fint itself, on a fine uniform mesh.
            original, _ = potential.radial_projector(angular_momentum, 1)
            transform = fourier_transform(potential, original, angular_momentum, count)
            mesh = np.linspace(0.3, 4.0, 38)
            step = 1e-4
            numerical = (
                filtered_function(mesh + step, transform, angular_momentum, spacing, alpha, beta1)
                - filtered_function(mesh - step, transform, angular_momentum, spacing, alpha, beta1)
            ) / (2 * step)
            analytic = filtered_derivative(mesh, transform, angular_momentum, spacing, alpha, beta1)
            scale = np.max(np.abs(analytic))
            self.assertGreater(scale, 0.1)
            np.testing.assert_allclose(analytic, numerical, atol=1e-6 * scale)
            # Not the projector itself (the previous tree's dvw = fint).
            self.assertGreater(np.max(np.abs(slope - table)), 0.1 * scale)

    def test_species_parameters(self) -> None:
        species = SpeciesPotential(DATA / "Si_POTRE.DAT", 1)
        self.assertFalse(species.has_fourier_filter)
        self.assertNotIn("fourier", repr(species))
        core_only = replace(species, fourier_filter_core=0.7)
        self.assertTrue(core_only.has_fourier_filter)
        potentials = {"Si": read_parsec_pseudopotential(DATA / "Si_POTRE.DAT")}
        result = apply_fourier_filters(potentials, {"Si": species}, 0.5)
        self.assertIs(result["Si"], potentials["Si"])
        filtered = apply_fourier_filters(
            potentials, {"Si": replace(species, fourier_filter_alpha=0.8)}, 0.5
        )
        self.assertEqual(sorted(filtered["Si"].filtered_projectors), [0, 2])
        with self.assertRaises(ValueError):
            SpeciesPotential(DATA / "Si_POTRE.DAT", 1, fourier_filter_beta1=-1.0)


class FilterInputTests(unittest.TestCase):
    PERIODIC = """Boundary_Conditions: bulk
Periodic_System: .true.
begin Cell_Shape
  10.2612  10.2612  10.2612
end Cell_Shape
Grid_Spacing: 0.51306
Expansion_Order: 8
States_Num: 22
Atom_Types_Num: 1
Atom_Type: Si
Local_Component: p
{extra}
begin Atom_Coord
  0.3 0.2 0.1
end Atom_Coord
"""
    CLUSTER = """Boundary_Conditions: cluster
Boundary_Sphere_Radius: 8.0
Grid_Spacing: 0.6
States_Num: 6
Atom_Types_Num: 1
Atom_Type: Si
Local_Component: p
{extra}
begin Atom_Coord
  0 0 0
end Atom_Coord
"""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        (self.directory / "Si_POTRE.DAT").write_bytes((DATA / "Si_POTRE.DAT").read_bytes())

    def tearDown(self) -> None:
        self._directory.cleanup()

    def parse(self, template: str, extra: str):
        path = self.directory / "parsec.in"
        path.write_text(template.format(extra=extra), encoding="utf-8")
        return parse_parsec_input(path)

    def test_periodic_and_cluster(self) -> None:
        translation = self.parse(self.PERIODIC, "Alpha_Filter: 0.8\nBeta1_Filter: 5.0")
        species = translation.problem.pseudopotentials["Si"]
        self.assertEqual(
            (species.fourier_filter_alpha, species.fourier_filter_beta1, species.fourier_filter_core),
            (0.8, 5.0, None),
        )
        self.assertTrue(species.has_fourier_filter)
        # The accelerated periodic path takes the readable preparation, which
        # filters (acceleration/tests/test_pseudopotential_setup.py).
        self.assertEqual(
            _reference_only_features(translation, accelerated_backend_selected=False), []
        )
        core = self.parse(self.PERIODIC, "Core_Filter: 0.6").problem.pseudopotentials["Si"]
        self.assertEqual((core.fourier_filter_alpha, core.fourier_filter_core), (0.0, 0.6))
        zero = self.parse(self.PERIODIC, "Alpha_Filter: 0").problem.pseudopotentials["Si"]
        self.assertFalse(zero.has_fourier_filter)
        with self.assertRaisesRegex(ParsecInputError, "zero or positive"):
            self.parse(self.PERIODIC, "Alpha_Filter: -0.5")
        # The lines of fourier_f.f90, only where a species is filtered.
        import io

        from parsec_python.Output.parsec_output import ParsecTextReporter

        stream = io.StringIO()
        ParsecTextReporter(stream.write, translation).header()
        self.assertIn(
            " Si     nql=  820     delq= 0.015\n"
            " Actually also performing fourier filtering in pseudopotentials \n"
            " Actually also performing fourier filtering in core charge \n"
            "  alpha =     0.8000  beta1 =     5.0000  acore =     0.8000\n",
            stream.getvalue(),
        )
        stream = io.StringIO()
        ParsecTextReporter(stream.write, self.parse(self.PERIODIC, "")).header()
        self.assertNotIn("Fourier", stream.getvalue())
        cluster = self.parse(self.CLUSTER, "Alpha_Filter: 0.8")
        self.assertFalse(cluster.problem.pseudopotentials["Si"].has_fourier_filter)
        self.assertTrue(any("periodic systems only" in note for note in cluster.warnings))
        quiet = self.parse(self.CLUSTER, "Alpha_Filter: 0")
        self.assertFalse(any("periodic systems only" in note for note in quiet.warnings))


if __name__ == "__main__":
    unittest.main()
