"""Old Martins pseudopotentials (``*_POTR.DAT``) against Fortran PARSEC.

The reader is compared with what ``parsec.out`` reports for the same files
(grid parameters, normalization, Kleinman-Bylander integrals, sizes of the
nonlocal blocks, charge of the superposed atomic density) and with the
new-format reader for a file rewritten in the old layout, on which Fortran
PARSEC gives the energy and forces of the new-format run.  The layout rules
of ``pseudo.f90`` and its refusals are tested on small synthetic files.
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from parsec_python.Input import (
    ParsecInputError,
    UnsupportedParsecOptionError,
    parse_parsec_input,
    summarize_translation,
)
from parsec_python.Pseudopotential import (
    parsec_radial_integral,
    read_martins_old_pseudopotential,
    read_parsec_pseudopotential,
)
from parsec_python.Pseudopotential.potr import logarithmic_grid_parameters
from parsec_python.Pseudopotential.radial_derivatives import valence_density_table
from parsec_python.SCF import prepare_single_point
from parsec_python.V_ion import (
    build_nonlocal_projectors,
    load_pseudopotentials,
    superpose_atomic_density,
)
from parsec_python.models import SpeciesPotential
from parsec_python.tests.martins_old_reference import (
    DATA,
    load_cases,
    support_sizes,
    write_case,
    write_martins_old,
)


def assert_printed(test, value: float, printed: float, digits: int) -> None:
    """``value`` rounds to ``printed``, a number with ``digits`` significant digits."""

    if printed == 0.0:
        test.assertAlmostEqual(value, 0.0, delta=0.5 * 10.0**-digits)
        return
    unit = 10.0 ** (math.floor(math.log10(abs(printed))) - digits + 1)
    test.assertAlmostEqual(value, printed, delta=0.5001 * unit)


def read_silicon(**changes):
    options = dict(
        symbol="Si",
        number_of_channels=2,
        electrons_per_orbital=(2.0, 2.0, 0.0),
        core_cutoff_radius=2.5,
    )
    options.update(changes)
    return read_martins_old_pseudopotential(DATA / "Si_POTR.DAT", **options)


class FortranReportTests(unittest.TestCase):
    """Numbers ``pseudo.f90`` prints for an old Martins file."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_cases()

    def assert_report(self, potential, reported: dict, local_l: int) -> None:
        # g13.6 prints six significant digits.
        assert_printed(self, potential.radial_parameter_a, reported["radial_parameter_a"], 6)
        assert_printed(self, potential.radial_parameter_b, reported["radial_parameter_b"], 6)
        self.assertEqual(potential.number_of_channels, reported["components"])
        self.assertEqual(
            [potential.channel_occupations[l] for l in sorted(potential.channel_occupations)],
            reported["occupancy"],
        )
        self.assertEqual(potential.has_nonlinear_core_correction, reported["core_correction"])
        # "Core radius [bohr] =" is printed with two decimals.
        self.assertAlmostEqual(potential.core_radius, reported["core_radius"], delta=0.00501)
        for angular_momentum in sorted(potential.radial_wavefunctions):
            wave = potential.radial_wavefunctions[angular_momentum]
            norm = parsec_radial_integral(potential.radii, wave * wave)
            # f15.8: eight decimals.
            self.assertAlmostEqual(
                norm, reported["normalization"][angular_momentum], delta=5.01e-9
            )
            integral = parsec_radial_integral(
                potential.radii,
                wave
                * wave
                * (
                    potential.channel_potentials[angular_momentum]
                    - potential.channel_potentials[local_l]
                ),
            )
            self.assertAlmostEqual(
                integral, reported["kb_integral"][angular_momentum], delta=5.01e-9
            )

    def test_silicon_file_of_the_relaxation_example(self) -> None:
        silicon = read_silicon()
        reported = self.cases["sina4"]["pseudopotentials"]["Si"]
        self.assertEqual(reported["format"], "martins_old")
        self.assert_report(silicon, reported, reported["local_angular_momentum"])
        self.assertEqual(silicon.file_format, "martins_old")
        self.assertEqual(silicon.symbol, "Si")
        self.assertEqual(silicon.radii.size, 1318)
        # zion is the sum of Electron_Per_Orbital; the third value of the
        # block lies beyond Potential_Num and is not read.
        self.assertEqual(silicon.ionic_charge, 4.0)
        self.assertEqual(silicon.channel_occupations, {0: 2.0, 1: 2.0})
        self.assertEqual(sorted(silicon.channel_potentials), [0, 1])
        self.assertFalse(np.any(silicon.core_density))
        # The file stores V in Rydberg, not r*V: first record of the file.
        self.assertEqual(silicon.radii[0], 0.1779419538e-05)
        self.assertEqual(silicon.channel_potentials[0][0], -0.3460094037e01)
        self.assertEqual(silicon.channel_potentials[1][0], -0.5778067056e01)
        # Ionic tail of the last record, -2*zion/r.
        self.assertAlmostEqual(
            silicon.channel_potentials[1][-1] * silicon.radii[-1], -8.0, delta=1e-6
        )

    def test_core_corrected_file_equals_its_new_format_original(self) -> None:
        """Ge rewritten in the old layout: Fortran reports it as the new one."""

        case = self.cases["ge2h2_old_format"]
        reported = case["pseudopotentials"]["Ge"]
        self.assertEqual(reported["format"], "martins_old")
        new = read_parsec_pseudopotential(DATA / "Ge_POTRE.DAT")
        occupations = [new.channel_occupations[l] for l in sorted(new.channel_occupations)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Ge_POTR.DAT"
            write_martins_old(new, path)
            old = read_martins_old_pseudopotential(
                path,
                symbol="Ge",
                number_of_channels=3,
                electrons_per_orbital=occupations,
                core_cutoff_radius=new.projector_support_radius,
            )
        self.assert_report(old, reported, reported["local_angular_momentum"])
        np.testing.assert_array_equal(old.radii, new.radii)
        for angular_momentum in (0, 1, 2):
            np.testing.assert_array_equal(
                old.channel_potentials[angular_momentum],
                new.channel_potentials[angular_momentum],
            )
            np.testing.assert_array_equal(
                old.radial_wavefunctions[angular_momentum],
                new.radial_wavefunctions[angular_momentum],
            )
        # The core charge is a volume density in the old layout.
        self.assertTrue(old.has_nonlinear_core_correction)
        self.assertGreater(float(old.core_density.max()), 0.1)
        np.testing.assert_array_equal(old.core_density, new.core_density)
        self.assertEqual(old.ionic_charge, new.ionic_charge)
        self.assertEqual(old.channel_occupations, new.channel_occupations)
        self.assertEqual(old.projector_support_radius, new.projector_support_radius)
        np.testing.assert_array_equal(
            old.valence_density, valence_density_table(new, read_from_file=False)
        )
        for angular_momentum in (0, 2):
            np.testing.assert_array_equal(
                old.radial_projector(angular_momentum, 1)[0],
                new.radial_projector(angular_momentum, 1)[0],
            )
        # The derived grid parameters are those of the file header to the
        # twelve digits the radii are stored with.
        self.assertAlmostEqual(
            old.radial_parameter_a / new.radial_parameter_a, 1.0, delta=1e-8
        )
        self.assertAlmostEqual(
            old.radial_parameter_b / new.radial_parameter_b, 1.0, delta=1e-8
        )

    def test_projector_spheres_and_atomic_density_on_the_grid(self) -> None:
        """Block sizes of nonloc.F90 and the charge initchrg.f90 reports."""

        for name, case in self.cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                translation = parse_parsec_input(write_case(case, Path(directory)))
                system = prepare_single_point(translation.problem)
                self.assertEqual(system.grid.size, case["hamiltonian_size"])
                self.assertEqual(support_sizes(system), case["nonlocal_block_sizes"])
                # Every stored projector value lies inside its sphere.
                matrix = system.nonlocal_operator.projectors
                for column, (atom, _, _) in enumerate(system.nonlocal_operator.labels):
                    rows = matrix.indices[matrix.indptr[column] : matrix.indptr[column + 1]]
                    self.assertGreater(rows.size, 0)
                    distance = np.linalg.norm(
                        system.grid.coordinates[rows] - system.atoms[atom].position, axis=1
                    )
                    potential = system.pseudopotentials[system.atoms[atom].symbol]
                    self.assertLessEqual(
                        float(distance.max()), potential.projector_support_radius
                    )
                charge = system.grid.integrate(
                    superpose_atomic_density(
                        system.grid,
                        system.atoms,
                        system.pseudopotentials,
                        translation.problem.pseudopotentials,
                    )
                )
                assert_printed(self, charge, case["atomic_density_charge"], 5)

    def test_core_radius_is_the_radius_of_the_input(self) -> None:
        """The two rules give different spheres where Fortran was run."""

        shifted = self.cases["sina4_shifted_core_1.9"]
        silicon = read_silicon(core_cutoff_radius=1.9)
        following = float(silicon.radii[np.searchsorted(silicon.radii, 1.9, side="right")])
        self.assertEqual(silicon.projector_support_radius, 1.9)
        with tempfile.TemporaryDirectory() as directory:
            translation = parse_parsec_input(write_case(shifted, Path(directory)))
            system = prepare_single_point(translation.problem)
        distance = np.linalg.norm(
            system.grid.coordinates - system.atoms[0].position, axis=1
        )
        self.assertEqual(int(np.count_nonzero(distance <= 1.9)), 54)
        self.assertEqual(shifted["nonlocal_block_sizes"][0], 54)
        # The rule of a new-format file, the next radial sample, would add four.
        self.assertEqual(int(np.count_nonzero(distance <= following)), 58)

    def test_support_radius_rules(self) -> None:
        hydrogen = read_parsec_pseudopotential(DATA / "H_POTRE.DAT")
        self.assertIsNone(hydrogen.core_radius)
        self.assertEqual(hydrogen.file_format, "martins_new")
        largest = max(hydrogen.channel_cutoffs.values())
        beyond = hydrogen.radii[hydrogen.radii > largest][0]
        self.assertEqual(hydrogen.projector_support_radius, float(beyond))
        # nonloc.F90 caps the radius at the penultimate radial sample.
        self.assertEqual(
            read_silicon(core_cutoff_radius=500.0).projector_support_radius,
            float(read_silicon().radii[-2]),
        )


class ProjectorBuilderGuardTests(unittest.TestCase):
    """A substituted builder must end the projectors at the core radius."""

    @staticmethod
    def next_sample_builder(grid, atoms, potentials, specifications):
        widened = {
            symbol: potential
            if potential.core_radius is None
            else replace(
                potential,
                core_radius=float(
                    potential.radii[
                        np.searchsorted(potential.radii, potential.core_radius, side="right")
                    ]
                ),
            )
            for symbol, potential in potentials.items()
        }
        return build_nonlocal_projectors(grid, atoms, widened, specifications)

    def prepare(self, name: str, builder):
        with tempfile.TemporaryDirectory() as directory:
            translation = parse_parsec_input(write_case(load_cases()[name], Path(directory)))
            return prepare_single_point(
                translation.problem, nonlocal_projector_builder=builder
            )

    def test_a_wider_sphere_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "beyond the core radius 1.900000 bohr"):
            self.prepare("sina4_shifted_core_1.9", self.next_sample_builder)

    def test_the_rule_of_the_input_passes(self) -> None:
        system = self.prepare("sina4_shifted_core_1.9", build_nonlocal_projectors)
        self.assertEqual(support_sizes(system)[0], 54)

    def test_points_without_projector_values_do_not_count(self) -> None:
        # At 2.5 bohr the Si projector has ended: the four grid points up to
        # the next radial sample carry zeros and are not stored.
        system = self.prepare("sina4_shifted", self.next_sample_builder)
        self.assertEqual(support_sizes(system)[0], 129)


def synthetic_tables(count: int = 260):
    """An exact logarithmic grid with two smooth channels."""

    parameter_a, parameter_b = 2.0e-4, 0.04
    radii = parameter_a * (np.exp(parameter_b * np.arange(1, count + 1)) - 1.0)
    potentials = [-2.0 * np.exp(-radii) - 1.0, -3.0 * np.exp(-0.5 * radii) + 0.25]
    waves = [radii * np.exp(-radii), radii * radii * np.exp(-0.7 * radii)]
    core = 0.3 * np.exp(-2.0 * radii)
    return parameter_a, parameter_b, radii, potentials, waves, core


def synthetic_file(
    path: Path,
    *,
    count: int = 260,
    header: str | None = None,
    slots: str = "1 2",
    columns: tuple[int, int] = (0, 1),
    core_rows: int | None = None,
    number: str = "{:.16E}",
    wrap: bool = False,
) -> None:
    _, _, radii, potentials, waves, core = synthetic_tables(count)
    lines = [header or f"{count} 2 {0 if core_rows is None else 1}", slots]
    for table in (potentials, waves):
        for row, radius in enumerate(radii):
            values = [number.format(table[column][row]) for column in columns]
            if wrap:
                # A record may continue on the next line, and text after the
                # last item of a list-directed read is not read.
                lines.append(f"{number.format(radius)} {values[0]}")
                lines.append(f"{values[1]}   trailing text")
            else:
                lines.append(" ".join([number.format(radius), *values]))
    if core_rows is not None:
        for row in range(core_rows):
            lines.append(f"{number.format(radii[row])} {number.format(core[row])}")
    path.write_text("\n".join(lines) + "\n")


class LayoutTests(unittest.TestCase):
    """Reading rules of the MARTINS_OLD branches of ``pseudo.f90``."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.path = Path(self._directory.name) / "X_POTR.DAT"

    def read(self, **changes):
        options = dict(
            symbol="X",
            number_of_channels=2,
            electrons_per_orbital=(1.5, 0.5),
            core_cutoff_radius=1.2,
        )
        options.update(changes)
        return read_martins_old_pseudopotential(self.path, **options)

    def test_grid_parameters_follow_pseudo_f90(self) -> None:
        parameter_a, parameter_b, radii, _, _, _ = synthetic_tables()
        derived_a, derived_b = logarithmic_grid_parameters(radii)
        self.assertAlmostEqual(derived_a / parameter_a, 1.0, delta=1e-11)
        self.assertAlmostEqual(derived_b / parameter_b, 1.0, delta=1e-11)
        # Fortran indices i_0 = 101 and j0 = 201: file samples 100 and 200.
        first, second = radii[99], radii[199]
        self.assertEqual(derived_a, first * first / (second - 2.0 * first))
        self.assertEqual(derived_b, float(np.log(second / derived_a + 1.0) / 200.0))
        logarithmic_grid_parameters(radii[:200])
        with self.assertRaisesRegex(ValueError, "at least 200 radial samples"):
            logarithmic_grid_parameters(radii[:199])

    def test_tables_and_derived_quantities(self) -> None:
        parameter_a, parameter_b, radii, potentials, waves, _ = synthetic_tables()
        synthetic_file(self.path)
        potential = self.read()
        np.testing.assert_array_equal(potential.radii, radii)
        for angular_momentum in (0, 1):
            np.testing.assert_array_equal(
                potential.channel_potentials[angular_momentum], potentials[angular_momentum]
            )
            np.testing.assert_array_equal(
                potential.radial_wavefunctions[angular_momentum], waves[angular_momentum]
            )
        self.assertAlmostEqual(potential.radial_parameter_a / parameter_a, 1.0, delta=1e-11)
        self.assertAlmostEqual(potential.radial_parameter_b / parameter_b, 1.0, delta=1e-11)
        self.assertEqual(potential.ionic_charge, 2.0)
        self.assertEqual(potential.channel_occupations, {0: 1.5, 1: 0.5})
        self.assertEqual(potential.core_radius, 1.2)
        self.assertEqual(potential.projector_support_radius, 1.2)
        self.assertFalse(potential.has_nonlinear_core_correction)
        np.testing.assert_allclose(
            potential.valence_density,
            (1.5 * waves[0] ** 2 + 0.5 * waves[1] ** 2) / (4.0 * np.pi * radii**2),
            rtol=1e-14,
        )

    def test_columns_follow_the_angular_momentum_list(self) -> None:
        _, _, _, potentials, waves, _ = synthetic_tables()
        synthetic_file(self.path, slots="2 1", columns=(1, 0))
        potential = self.read()
        np.testing.assert_array_equal(potential.channel_potentials[0], potentials[0])
        np.testing.assert_array_equal(potential.channel_potentials[1], potentials[1])
        np.testing.assert_array_equal(potential.radial_wavefunctions[1], waves[1])
        # Occupations follow the angular momentum, not the column.
        self.assertEqual(potential.channel_occupations, {0: 1.5, 1: 0.5})

    def test_list_directed_records(self) -> None:
        synthetic_file(self.path)
        plain = self.read()
        synthetic_file(self.path, wrap=True, number="{:.16E}")
        wrapped = self.read()
        np.testing.assert_array_equal(wrapped.radii, plain.radii)
        np.testing.assert_array_equal(
            wrapped.channel_potentials[1], plain.channel_potentials[1]
        )
        np.testing.assert_array_equal(
            wrapped.radial_wavefunctions[0], plain.radial_wavefunctions[0]
        )
        # Fortran double-precision exponents.
        self.path.write_text(self.path.read_text().replace("E", "D"))
        exponent_d = self.read()
        np.testing.assert_array_equal(exponent_d.radii, plain.radii)
        np.testing.assert_array_equal(
            exponent_d.channel_potentials[0], plain.channel_potentials[0]
        )

    def test_core_charge_is_a_volume_density_and_may_end_early(self) -> None:
        _, _, radii, _, _, core = synthetic_tables()
        synthetic_file(self.path, core_rows=radii.size)
        complete = self.read()
        self.assertTrue(complete.has_nonlinear_core_correction)
        np.testing.assert_array_equal(complete.core_density, core)
        # read(20,*,end=80): the rest of a short block stays zero.
        synthetic_file(self.path, core_rows=40)
        partial = self.read()
        np.testing.assert_array_equal(partial.core_density[:40], core[:40])
        self.assertFalse(np.any(partial.core_density[40:]))
        # icore = 0: a core block is not read at all.
        synthetic_file(self.path, core_rows=radii.size, header=f"{radii.size} 2 0")
        self.assertFalse(np.any(self.read().core_density))
        self.assertFalse(self.read().has_nonlinear_core_correction)

    def test_refusals(self) -> None:
        synthetic_file(self.path)
        # pseudo.f90 error 202.
        with self.assertRaisesRegex(ValueError, "holds 2 pseudopotential components"):
            self.read(number_of_channels=3, electrons_per_orbital=(1, 1, 0))
        # usrinputfile.F90 error 138.
        for channels in (0, 5):
            with self.assertRaisesRegex(ValueError, "impossible number of potentials"):
                self.read(number_of_channels=channels, electrons_per_orbital=(1,) * 5)
        # The list-directed read of the block ends before Potential_Num values.
        with self.assertRaisesRegex(ValueError, "Electron_Per_Orbital"):
            self.read(electrons_per_orbital=(2.0,))
        for radius in (0.0, -1.0, float("nan")):
            with self.assertRaisesRegex(ValueError, "Core_Cutoff_Radius must be positive"):
                self.read(core_cutoff_radius=radius)
        # pseudo.f90 error 206.
        synthetic_file(self.path, count=150)
        with self.assertRaisesRegex(ValueError, "at least 200 radial samples"):
            self.read()
        # Components PARSEC would not use.
        synthetic_file(self.path, slots="1 3")
        with self.assertRaisesRegex(ValueError, "angular-momentum list"):
            self.read()
        synthetic_file(self.path, slots="1 1")
        with self.assertRaisesRegex(ValueError, "angular-momentum list"):
            self.read()
        synthetic_file(self.path)
        lines = self.path.read_text().splitlines()
        self.path.write_text("\n".join(lines[:400]) + "\n")
        with self.assertRaisesRegex(ValueError, "ends before the pseudo-wavefunctions"):
            self.read()
        self.path.write_text("\n".join(lines[:10] + ["0.1 abc 0.2"] + lines[11:]) + "\n")
        with self.assertRaisesRegex(ValueError, "expected a number"):
            self.read()
        self.path.write_text("\n".join(["260 2.0 0"] + lines[1:]) + "\n")
        with self.assertRaisesRegex(ValueError, "expected an integer in the header"):
            self.read()
        # Blocks on different radial grids.
        shifted = [lines[0], lines[1]]
        for line in lines[2:262]:
            fields = line.split()
            shifted.append(" ".join([f"{float(fields[0]) * 1.01:.16E}", *fields[1:]]))
        self.path.write_text("\n".join(shifted + lines[262:]) + "\n")
        with self.assertRaisesRegex(ValueError, "radial grid of the pseudopotentials differs"):
            self.read()


class SpeciesInputTests(unittest.TestCase):
    """What the model and the loader require of a martins_old species."""

    def test_species_model(self) -> None:
        species = SpeciesPotential(
            DATA / "Si_POTR.DAT",
            1,
            pseudopotential_format="Martins_Old",
            number_of_channels=2,
            electrons_per_orbital=(2, 2, 0),
            core_cutoff_radius=2.5,
        )
        self.assertEqual(species.pseudopotential_format, "martins_old")
        self.assertEqual(species.electrons_per_orbital, (2.0, 2.0))
        self.assertEqual(SpeciesPotential("H_POTRE.DAT", 0).pseudopotential_format, "martins_new")
        with self.assertRaisesRegex(ValueError, "needs number_of_channels"):
            SpeciesPotential("Si_POTR.DAT", 1, pseudopotential_format="martins_old")
        with self.assertRaisesRegex(ValueError, "belong to a martins_old"):
            SpeciesPotential("H_POTRE.DAT", 0, core_cutoff_radius=2.5)
        with self.assertRaisesRegex(ValueError, "martins_new' or 'martins_old"):
            SpeciesPotential("X_FHIPP.DAT", 0, pseudopotential_format="fhipp")
        old = dict(
            pseudopotential_format="martins_old",
            number_of_channels=2,
            electrons_per_orbital=(2, 2),
            core_cutoff_radius=2.5,
        )
        # usrinputfile.F90 error 139: local component beyond the channels.
        with self.assertRaisesRegex(ValueError, "beyond the number of channels"):
            SpeciesPotential("Si_POTR.DAT", 2, **old)
        with self.assertRaisesRegex(ValueError, "1 to 4"):
            SpeciesPotential("Si_POTR.DAT", 0, **{**old, "number_of_channels": 5})
        with self.assertRaisesRegex(ValueError, "every channel"):
            SpeciesPotential("Si_POTR.DAT", 0, **{**old, "electrons_per_orbital": (2,)})
        with self.assertRaisesRegex(ValueError, "must be positive"):
            SpeciesPotential("Si_POTR.DAT", 0, **{**old, "core_cutoff_radius": 0.0})

    def test_mixed_formats_are_loaded_together(self) -> None:
        species = {
            "Si": SpeciesPotential(
                DATA / "Si_POTR.DAT",
                1,
                pseudopotential_format="martins_old",
                number_of_channels=2,
                electrons_per_orbital=(2, 2),
                core_cutoff_radius=2.5,
            ),
            "Na": SpeciesPotential(DATA / "Na_POTRE.DAT", 1),
        }
        potentials = load_pseudopotentials(species)
        self.assertEqual(potentials["Si"].file_format, "martins_old")
        self.assertEqual(potentials["Si"].core_radius, 2.5)
        self.assertEqual(potentials["Na"].file_format, "martins_new")
        self.assertIsNone(potentials["Na"].core_radius)
        # An old file names no functional: it is accepted for PBE as well.
        self.assertEqual(
            load_pseudopotentials({"Si": species["Si"]}, xc_functional="pbe")["Si"].correlation,
            "",
        )


class InputTranslationTests(unittest.TestCase):
    """Per-species keywords of ``usrinputfile.F90`` for the old format."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        self.case = load_cases()["sina4"]
        self.lines = list(self.case["input"])

    def parse(self, lines=None):
        case = dict(self.case, input=self.lines if lines is None else lines)
        return parse_parsec_input(write_case(case, self.directory))

    def species_blocks(self):
        """The Si lines, the Na lines and what precedes and follows them."""

        first = self.lines.index("Atom_Type: Si")
        second = self.lines.index("Atom_Type: Na")
        end = max(i for i, line in enumerate(self.lines) if line.startswith("end Atom_Coord"))
        return (
            self.lines[:first],
            self.lines[first:second],
            self.lines[second : end + 1],
            self.lines[end + 1 :],
        )

    def replaced(self, old: str, new: str | None):
        index = next(i for i, line in enumerate(self.lines) if line.startswith(old))
        lines = list(self.lines)
        if new is None:
            del lines[index]
        else:
            lines[index] = new
        return lines

    def test_species_of_the_example(self) -> None:
        translation = self.parse()
        silicon = translation.problem.pseudopotentials["Si"]
        self.assertEqual(silicon.pseudopotential_format, "martins_old")
        self.assertEqual(silicon.path.name, "Si_POTR.DAT")
        self.assertEqual(silicon.number_of_channels, 2)
        # The block holds "2 2 0"; Potential_Num values are read.
        self.assertEqual(silicon.electrons_per_orbital, (2.0, 2.0))
        self.assertEqual(silicon.core_cutoff_radius, 2.5)
        self.assertEqual(silicon.local_angular_momentum, 1)
        sodium = translation.problem.pseudopotentials["Na"]
        self.assertEqual(sodium.pseudopotential_format, "martins_new")
        self.assertEqual(sodium.path.name, "Na_POTRE.DAT")
        self.assertIsNone(sodium.core_cutoff_radius)
        summary = summarize_translation(translation)
        self.assertIn("martins_old: Potential_Num=2", summary)
        self.assertIn("Core_Cutoff_Radius=2.5 bohr", summary)

    def test_unmodified_input_of_the_fortran_example(self) -> None:
        lines = (DATA / "sina4_relaxation_parsec.in").read_text().splitlines()
        translation = self.parse(lines)
        problem = translation.problem
        self.assertEqual([atom.symbol for atom in problem.atoms], ["Si"] + ["Na"] * 4)
        self.assertAlmostEqual(problem.grid.radius, 4.0 / 0.529177, delta=1e-12)
        self.assertEqual(problem.grid.spacing, 0.8)
        self.assertEqual(problem.scf.number_of_states, 8)
        self.assertEqual(problem.scf.max_iterations, 50)
        self.assertEqual(problem.scf.fixed_occupations, (1.0,) * 4 + (0.0,) * 4)
        self.assertEqual(problem.eigensolver.method, "chebff")
        self.assertEqual(problem.mixing.parameter, 0.28)
        self.assertEqual(problem.mixing.memory, 3)
        self.assertEqual(translation.relaxation.method, "bfgs")
        self.assertEqual(translation.relaxation.max_movements, 10)
        self.assertEqual(translation.relaxation.minimum_force, 0.03)
        # Move_Flag: 0 for Si and no Move_Flag for Na: every atom moves.
        self.assertEqual(translation.movable, (True,) * 5)

    def test_core_cutoff_radius_takes_a_length_unit(self) -> None:
        # Fortran reports "Core radius [bohr] =  2.50" for this line.
        translation = self.parse(
            self.replaced("Core_Cutoff_Radius", "Core_Cutoff_Radius: 1.3229425 ang")
        )
        self.assertAlmostEqual(
            translation.problem.pseudopotentials["Si"].core_cutoff_radius, 2.5, delta=1e-9
        )

    def test_potential_num_is_handed_out_in_species_order(self) -> None:
        head, silicon, sodium, tail = self.species_blocks()
        # Na first: its read takes the only Potential_Num and Fortran stops
        # with error 138, "impossible number of potentials", for Si.
        with self.assertRaisesRegex(ParsecInputError, "n-th Potential_Num"):
            self.parse(head + sodium + silicon + tail)
        # With a line for Na to take, Fortran gives the energy of the example.
        translation = self.parse(
            head + [sodium[0], "Potential_Num: 3"] + sodium[1:] + silicon + tail
        )
        self.assertEqual(list(translation.problem.pseudopotentials), ["Na", "Si"])
        self.assertEqual(translation.problem.pseudopotentials["Si"].number_of_channels, 2)
        self.assertIsNone(translation.problem.pseudopotentials["Na"].number_of_channels)

    def test_missing_or_misplaced_species_input_is_refused(self) -> None:
        # Fortran would run without the nonlocal projectors of Si.
        with self.assertRaisesRegex(ParsecInputError, "needs Core_Cutoff_Radius"):
            self.parse(self.replaced("Core_Cutoff_Radius", None))
        with self.assertRaisesRegex(ParsecInputError, "Core_Cutoff_Radius must be positive"):
            self.parse(self.replaced("Core_Cutoff_Radius", "Core_Cutoff_Radius: 0"))
        with self.assertRaisesRegex(ParsecInputError, "needs Potential_Num"):
            self.parse(self.replaced("Potential_Num", None))
        for count in (0, 5):
            with self.assertRaisesRegex(ParsecInputError, "impossible number of potentials"):
                self.parse(self.replaced("Potential_Num", f"Potential_Num: {count}"))
        # usrinputfile.F90 error 139.
        with self.assertRaisesRegex(ParsecInputError, "beyond the number of channels"):
            lines = self.replaced("Local_Component: p", "Local_Component: d")
            self.parse(lines)
        # A line under the new-format species would go to the next old one.
        head, silicon, sodium, tail = self.species_blocks()
        with self.assertRaisesRegex(ParsecInputError, "core_cutoff_radius occurs 2 times"):
            self.parse(
                head + silicon + [sodium[0], "Core_Cutoff_Radius: 3.0"] + sodium[1:] + tail
            )
        # usrinputfile.F90 error 234.
        with self.assertRaisesRegex(ParsecInputError, "Read_VCD needs a martins_new"):
            self.parse(head + [silicon[0], "Read_VCD: .true."] + silicon[1:] + sodium + tail)

    def test_read_vcd_beside_an_old_format_species(self) -> None:
        head, silicon, sodium, tail = self.species_blocks()
        with_vcd = [sodium[0], "Read_VCD: .true."] + sodium[1:]
        # The only Read_VCD line goes to the first Atom_Type, Si: Fortran
        # stops with error 234 of usrinputfile.F90.
        with self.assertRaisesRegex(ParsecInputError, "Read_VCD needs a martins_new"):
            self.parse(head + silicon + with_vcd + tail)
        # With a line for Si to take, Fortran runs and gives the energy of
        # the example: the file of Na holds the density of its orbitals.
        translation = self.parse(
            head + [silicon[0], "Read_VCD: .false."] + silicon[1:] + with_vcd + tail
        )
        self.assertFalse(translation.problem.pseudopotentials["Si"].read_valence_density)
        self.assertTrue(translation.problem.pseudopotentials["Na"].read_valence_density)
        # Na first and Si last: pseudo.f90 tests the format of the last
        # Atom_Type for the Read_VCD of Na and stops with error 23.
        with self.assertRaisesRegex(ParsecInputError, "tests the format of the last Atom_Type"):
            self.parse(
                head + [with_vcd[0], "Potential_Num: 3"] + with_vcd[1:] + silicon + tail
            )

    def test_electron_per_orbital_block(self) -> None:
        block = self.lines.index("begin Electron_Per_Orbital")
        end = self.lines.index("end Electron_Per_Orbital")
        without = self.lines[:block] + self.lines[end + 1 :]
        # Fortran goes on with zero occupations, hence a zero ionic charge.
        with self.assertRaisesRegex(ParsecInputError, "needs an Electron_Per_Orbital block"):
            self.parse(without)
        # usrinputfile.F90 error 137.
        with self.assertRaisesRegex(ParsecInputError, "only one line"):
            self.parse(self.lines[:end] + ["0 0"] + self.lines[end:])
        # Fortran ends with a read error.
        with self.assertRaisesRegex(ParsecInputError, "occupations of the 2 channels"):
            self.parse(self.replaced("2 2 0", "4"))
        translation = self.parse(self.replaced("2 2 0", "1.5 2.5"))
        self.assertEqual(
            translation.problem.pseudopotentials["Si"].electrons_per_orbital, (1.5, 2.5)
        )

    def test_potential_num_must_agree_with_the_file(self) -> None:
        # pseudo.f90 error 202, raised when the file is read.
        translation = self.parse(self.replaced("Potential_Num", "Potential_Num: 3"))
        self.assertEqual(
            translation.problem.pseudopotentials["Si"].electrons_per_orbital, (2.0, 2.0, 0.0)
        )
        with self.assertRaisesRegex(ValueError, "Potential_Num is 3"):
            load_pseudopotentials(translation.problem.pseudopotentials)

    def test_other_formats_and_missing_files(self) -> None:
        for name in ("martins_wang", "fhipp"):
            with self.assertRaisesRegex(UnsupportedParsecOptionError, "martins_new or martins_old"):
                self.parse(
                    self.replaced(
                        "Pseudopotential_Format: martins_old",
                        f"Pseudopotential_Format: {name}",
                    )
                )
        translation_path = write_case(self.case, self.directory)
        (self.directory / "Si_POTR.DAT").unlink()
        with self.assertRaisesRegex(ParsecInputError, "Si_POTR.DAT"):
            parse_parsec_input(translation_path)


if __name__ == "__main__":
    unittest.main()
