"""Martins-Wang and FHIPP pseudopotentials and the per-species input labels.

The readers are tested on small synthetic files for the layout rules of
``pseudo.f90`` and on the files of the Fortran examples against what Fortran
PARSEC prints for them (grid parameters, normalization, Kleinman-Bylander
integrals, sizes of the nonlocal blocks).  The input tests cover the ESDF
cursors of the per-species labels (``usrinputfile.F90:1756-1907``): the n-th
occurrence of a label goes to the n-th species that reads it.
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from parsec_python.Grid import build_cluster_grid
from parsec_python.Input import (
    ParsecInputError,
    UnsupportedParsecOptionError,
    parse_parsec_input,
)
from parsec_python.Pseudopotential import (
    parsec_radial_integral,
    read_fhipp_pseudopotential,
    read_martins_wang_header,
    read_martins_wang_pseudopotential,
    read_parsec_pseudopotential,
)
from parsec_python.Pseudopotential.radial_derivatives import (
    local_potential_derivative_table,
    log_grid_derivative,
)
from parsec_python.V_ion import build_nonlocal_projectors, load_pseudopotentials
from parsec_python.models import Atom, GridSettings, SpeciesPotential

DATA = Path(__file__).resolve().parents[1] / "data"


def assert_printed(test, value: float, printed: float, digits: int) -> None:
    """``value`` rounds to ``printed``, which has ``digits`` significant digits."""

    if printed == 0.0:
        test.assertAlmostEqual(value, 0.0, delta=0.5 * 10.0**-digits)
        return
    unit = 10.0 ** (math.floor(math.log10(abs(printed))) - digits + 1)
    test.assertAlmostEqual(value, printed, delta=0.5001 * unit)


def kb_integral(potential, angular_momentum: int, local_l: int) -> float:
    wave = potential.radial_wavefunctions[angular_momentum]
    return parsec_radial_integral(
        potential.radii,
        wave
        * wave
        * (
            potential.channel_potentials[angular_momentum]
            - potential.channel_potentials[local_l]
        ),
    )


def martins_wang_grid(count: int = 240) -> np.ndarray:
    """Radii ``a*(exp(b*(i-1)) - 1)`` at Fortran indices ``i = 2..count+1``."""

    a, b = 2.5e-4, 0.05
    return a * (np.exp(b * np.arange(1, count + 1)) - 1.0)


def write_martins_wang(
    path: Path,
    *,
    references=(1, 1, 0),
    local_slot: int = 2,
    core: bool = False,
    count: int = 240,
    occupations=(1.0, 0.5, 0.0),
) -> np.ndarray:
    radii = martins_wang_grid(count)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(
            f"{count}, {int(core)}, 11,  1.5, {local_slot},  {occupations[0]},  "
            f"{occupations[1]},  {occupations[2]}, 0  |nrr,icor,iatom,z,loc\n"
        )
        stream.write(" ".join(str(value) for value in references) + "   1 1 0   |<-iref\n")
        for r in radii:
            values = [
                -1.0 / np.sqrt(r * r + 1.0),
                -0.8 / np.sqrt(r * r + 0.5),
                -0.6 / np.sqrt(r * r + 0.2),
                np.exp(-r),
                r * np.exp(-r),
                r * r * np.exp(-r),
            ]
            if core:
                values.append(0.3 * np.exp(-2.0 * r))
            # Two further columns (spin-orbit) are not read.
            values += [9.0, 9.0]
            stream.write(" ".join(f"{value:.12E}" for value in [r, *values]) + "\n")
        stream.write(" #r, vs(Hart.),vp,vd,phi_s,phi_p,phi_d\n")
    return radii


def write_fhipp(
    path: Path,
    *,
    channels: int = 3,
    mesh: int = 300,
    amesh: float = 1.0247,
    core: bool = True,
    file_channels: int | None = None,
    second_mesh: int | None = None,
    second_amesh: float | None = None,
) -> np.ndarray:
    radii = 4.0e-4 * amesh ** np.arange(1, mesh + 1)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"0.30000000000000E+01   {file_channels or channels}\n")
        for _ in range(10):
            stream.write("  0.0000    .00e+00   .00e+00\n")
        for l in range(channels):
            size = second_mesh if (l == 1 and second_mesh) else mesh
            ratio = second_amesh if (l == 1 and second_amesh) else amesh
            stream.write(f"{size}  {ratio:.14E}\n")
            for index, r in enumerate(radii[:size], start=1):
                u = r ** (l + 1) * np.exp(-r)
                v = -3.0 / np.sqrt(r * r + 0.3 * (l + 1))
                stream.write(f"{index:4d} {r:.14E} {u:.14E} {v:.14E}\n")
        if core:
            for r in radii:
                rho = 4.0 * np.pi * 0.2 * np.exp(-3.0 * r)
                stream.write(f"{r:.14E} {rho:.14E} {-rho:.14E} {rho:.14E}\n")
    return radii


class MartinsWangReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_layout_of_a_synthetic_file(self) -> None:
        path = self.directory / "X_POTRW.DAT"
        radii = write_martins_wang(path, core=True)
        header = read_martins_wang_header(path)
        self.assertEqual(header.present, (0, 1))
        self.assertEqual(header.local_angular_momentum, 1)
        potential = read_martins_wang_pseudopotential(path, symbol="X", core_cutoff_radius=2.0)
        np.testing.assert_allclose(potential.radii, radii, rtol=1e-11)
        self.assertEqual(sorted(potential.channel_potentials), [0, 1])
        # Hartree -> Rydberg, R_l -> u_l = r R_l, core as read.
        np.testing.assert_allclose(
            potential.channel_potentials[0], -2.0 / np.sqrt(radii**2 + 1.0), rtol=1e-11
        )
        np.testing.assert_allclose(
            potential.radial_wavefunctions[1], radii * radii * np.exp(-radii), rtol=1e-11
        )
        np.testing.assert_allclose(potential.core_density, 0.3 * np.exp(-2.0 * radii), rtol=1e-11)
        self.assertTrue(potential.has_nonlinear_core_correction)
        self.assertEqual(potential.ionic_charge, 1.5)
        self.assertEqual(potential.channel_occupations, {0: 1.0, 1: 0.5})
        self.assertEqual(potential.local_channel, 1)
        self.assertEqual(potential.core_radius, 2.0)
        self.assertEqual(potential.projector_support_radius, 2.0)
        self.assertEqual(potential.file_format, "martins_wang")
        # The grid parameters of pseudo.f90 (c = a).
        self.assertAlmostEqual(potential.radial_parameter_a / 2.5e-4, 1.0, delta=1e-10)
        self.assertAlmostEqual(potential.radial_parameter_b / 0.05, 1.0, delta=1e-10)
        self.assertEqual(potential.radial_offset, potential.radial_parameter_a)
        expected = (
            1.0 * (radii * np.exp(-radii)) ** 2 + 0.5 * (radii * radii * np.exp(-radii)) ** 2
        ) / (4.0 * np.pi * radii * radii)
        np.testing.assert_allclose(potential.valence_density, expected, rtol=1e-11)

    def test_refusals(self) -> None:
        path = self.directory / "X_POTRW.DAT"
        # Channels that are not s, s+p or s+p+d are mislabeled by pseudo.f90.
        write_martins_wang(path, references=(1, 0, 1))
        with self.assertRaisesRegex(ValueError, "must be s, s and p"):
            read_martins_wang_header(path)
        write_martins_wang(path, local_slot=0)
        with self.assertRaisesRegex(ValueError, "loc=0"):
            read_martins_wang_header(path)
        write_martins_wang(path, local_slot=3)
        with self.assertRaisesRegex(ValueError, "loc=3"):
            read_martins_wang_header(path)
        # pseudo.f90 error 210.
        write_martins_wang(path, count=150)
        with self.assertRaisesRegex(ValueError, "at least 200"):
            read_martins_wang_header(path)
        write_martins_wang(path)
        with self.assertRaisesRegex(ValueError, "positive"):
            read_martins_wang_pseudopotential(path, symbol="X", core_cutoff_radius=0.0)

    def test_sodium_file_against_fortran(self) -> None:
        # parsec.out of the NaH parity run (Fortran PARSEC, 2026-10-27).
        potential = read_martins_wang_pseudopotential(
            DATA / "Na_POTRW.DAT", symbol="Na", core_cutoff_radius=3.1
        )
        assert_printed(self, potential.radial_parameter_a, 0.225341e-03, 6)
        assert_printed(self, potential.radial_parameter_b, 0.100000e-01, 6)
        assert_printed(self, potential.radial_offset, 0.225341e-03, 6)
        self.assertFalse(potential.has_nonlinear_core_correction)
        self.assertEqual(potential.local_channel, 1)
        self.assertEqual(potential.number_of_channels, 3)
        self.assertEqual(potential.ionic_charge, 1.0)
        self.assertEqual(potential.channel_occupations, {0: 1.0, 1: 0.0, 2: 0.0})
        for l, norm in ((0, 1.00001667), (1, 1.00001667), (2, 1.01036503)):
            wave = potential.radial_wavefunctions[l]
            self.assertAlmostEqual(
                parsec_radial_integral(potential.radii, wave * wave), norm, delta=5.01e-9
            )
        for l, value in ((0, 0.01109253), (1, 0.0), (2, -0.0)):
            self.assertAlmostEqual(kb_integral(potential, l, 1), value, delta=5.01e-9)
        # "Sizes of all non-local blocks": 676 points for Na of the NaH case.
        grid = build_cluster_grid(GridSettings(spacing=0.3 / 0.529177, radius=6.0 / 0.529177))
        species = {
            "Na": SpeciesPotential(
                DATA / "Na_POTRW.DAT",
                1,
                pseudopotential_format="martins_wang",
                core_cutoff_radius=3.1,
            )
        }
        potentials = load_pseudopotentials(species)
        # The recentred position of Na in that run.  Fortran counts the grid
        # points within the projector radius; a column drops the points where
        # its value is zero (beyond the cutoffs of both channels).
        position = np.array([0.0, 0.0, -1.782957])
        radius = potentials["Na"].projector_support_radius
        inside = np.linalg.norm(grid.coordinates - position, axis=1) <= radius
        self.assertEqual(int(np.count_nonzero(inside)), 676)
        operator = build_nonlocal_projectors(grid, [Atom("Na", position)], potentials, species)
        rows = np.unique(operator.projectors.indices)
        self.assertTrue(set(rows.tolist()) <= set(np.flatnonzero(inside).tolist()))

    def test_loader_checks_the_local_channel(self) -> None:
        species = SpeciesPotential(
            DATA / "Na_POTRW.DAT", 0, pseudopotential_format="martins_wang", core_cutoff_radius=3.1
        )
        with self.assertRaisesRegex(ValueError, "names the local channel l=1"):
            load_pseudopotentials({"Na": species})
        with self.assertRaisesRegex(ValueError, "SO_PSP is not supported"):
            load_pseudopotentials({"Na": replace(species, local_angular_momentum=1, spin_orbit=True)})
        with self.assertRaisesRegex(ValueError, "only core_cutoff_radius"):
            SpeciesPotential(
                DATA / "Na_POTRW.DAT",
                1,
                pseudopotential_format="martins_wang",
                core_cutoff_radius=3.1,
                number_of_channels=3,
            )


class FHIPPReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def read(self, path: Path, **changes):
        options = dict(
            symbol="X",
            number_of_channels=3,
            electrons_per_orbital=(2.0, 1.0, 0.0, 0.0),
            core_cutoff_radius=2.2,
            nonlinear_core_correction=True,
        )
        options.update(changes)
        return read_fhipp_pseudopotential(path, **options)

    def test_layout_of_a_synthetic_file(self) -> None:
        path = self.directory / "X_FHIPP.DAT"
        radii = write_fhipp(path)
        potential = self.read(path)
        np.testing.assert_allclose(potential.radii, radii, rtol=1e-13)
        self.assertEqual(potential.ionic_charge, 3.0)
        self.assertEqual(potential.radial_parameter_c, 0.0)
        self.assertEqual(potential.radial_offset, 0.0)
        self.assertAlmostEqual(potential.radial_parameter_a, radii[0] / 1.0247, delta=1e-18)
        self.assertAlmostEqual(potential.radial_parameter_b, math.log(1.0247), delta=1e-15)
        np.testing.assert_allclose(
            potential.channel_potentials[2], -6.0 / np.sqrt(radii**2 + 0.9), rtol=1e-12
        )
        np.testing.assert_allclose(potential.radial_wavefunctions[1], radii**2 * np.exp(-radii), rtol=1e-12)
        # The core column is 4*pi times the volume density.
        np.testing.assert_allclose(potential.core_density, 0.2 * np.exp(-3.0 * radii), rtol=1e-12)
        self.assertEqual(potential.channel_occupations, {0: 2.0, 1: 1.0, 2: 0.0})
        self.assertEqual(potential.core_radius, 2.2)
        self.assertEqual(potential.file_format, "fhipp")
        without = self.read(path, nonlinear_core_correction=False)
        self.assertFalse(without.has_nonlinear_core_correction)
        np.testing.assert_array_equal(without.core_density, 0.0)

    def test_refusals(self) -> None:
        path = self.directory / "X_FHIPP.DAT"
        # pseudo.f90 error 203.
        write_fhipp(path, file_channels=4)
        with self.assertRaisesRegex(ValueError, "Potential_Num is 3"):
            self.read(path)
        # pseudo.f90 error 211.
        write_fhipp(path, second_mesh=299)
        with self.assertRaisesRegex(ValueError, "grid size"):
            self.read(path)
        # The check pseudo.f90 error 212 means.
        write_fhipp(path, second_amesh=1.0248)
        with self.assertRaisesRegex(ValueError, "radial increment"):
            self.read(path)
        write_fhipp(path, core=False)
        with self.assertRaisesRegex(ValueError, "Nonlinear_Core_Correction"):
            self.read(path)
        with self.assertRaisesRegex(ValueError, "Electron_Per_Orbital"):
            self.read(path, electrons_per_orbital=(2.0,), nonlinear_core_correction=False)

    def test_files_against_fortran(self) -> None:
        # parsec.out of the SiH4 and Mg2 parity runs (Fortran PARSEC, 2026-10-27).
        silicon = read_fhipp_pseudopotential(
            DATA / "Si_FHIPP.DAT",
            symbol="Si",
            number_of_channels=3,
            electrons_per_orbital=(2, 2, 0),
            core_cutoff_radius=2.5,
        )
        assert_printed(self, silicon.radial_parameter_a, 0.435668e-03, 6)
        assert_printed(self, silicon.radial_parameter_b, 0.243999e-01, 6)
        self.assertEqual(silicon.ionic_charge, 4.0)
        for l, norm, kb in ((0, 1.00009923, 0.73656875), (1, 1.00009923, 0.26759002), (2, 1.05091999, 0.0)):
            wave = silicon.radial_wavefunctions[l]
            self.assertAlmostEqual(parsec_radial_integral(silicon.radii, wave * wave), norm, delta=5.01e-9)
            self.assertAlmostEqual(kb_integral(silicon, l, 2), kb, delta=5.01e-9)
        magnesium = read_fhipp_pseudopotential(
            DATA / "Mg_FHIPP.DAT",
            symbol="Mg",
            number_of_channels=4,
            electrons_per_orbital=(2, 0, 0, 0),
            core_cutoff_radius=2.5,
            nonlinear_core_correction=True,
        )
        assert_printed(self, magnesium.radial_parameter_a, 0.508279e-03, 6)
        self.assertTrue(magnesium.has_nonlinear_core_correction)
        for l, norm, kb in (
            (0, 1.00009923, 0.25153067),
            (1, 1.04106403, 0.48964078),
            (2, 1.00009923, 0.0),
            (3, 1.09397018, -0.00289893),
        ):
            wave = magnesium.radial_wavefunctions[l]
            self.assertAlmostEqual(parsec_radial_integral(magnesium.radii, wave * wave), norm, delta=5.01e-9)
            self.assertAlmostEqual(kb_integral(magnesium, l, 2), kb, delta=5.01e-9)
        # "Sizes of all non-local blocks": 360 points for Si of SiH4 and each Mg.
        grid = build_cluster_grid(GridSettings(spacing=0.3 / 0.529177, radius=6.0 / 0.529177))
        species = {
            "Si": SpeciesPotential(
                DATA / "Si_FHIPP.DAT",
                2,
                pseudopotential_format="fhipp",
                number_of_channels=3,
                electrons_per_orbital=(2, 2, 0),
                core_cutoff_radius=2.5,
            )
        }
        radius = load_pseudopotentials(species)["Si"].projector_support_radius
        inside = np.linalg.norm(grid.coordinates, axis=1) <= radius
        self.assertEqual(int(np.count_nonzero(inside)), 360)


class RadialOffsetTests(unittest.TestCase):
    """``c`` of ``r(i) = a*exp(b*(i-1)) - c`` in the radial derivatives."""

    def test_martins_files_keep_c_equal_to_a(self) -> None:
        potential = read_parsec_pseudopotential(DATA / "spin_ca_O_POTRE.DAT")
        self.assertIsNone(potential.radial_parameter_c)
        self.assertEqual(potential.radial_offset, potential.radial_parameter_a)

    def test_derivative_on_an_fhipp_grid(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "X_FHIPP.DAT"
            write_fhipp(path, mesh=500, core=False)
            potential = read_fhipp_pseudopotential(
                path,
                symbol="X",
                number_of_channels=3,
                electrons_per_orbital=(2, 1, 0),
                core_cutoff_radius=2.0,
            )
        radii = potential.radii
        derivative = log_grid_derivative(potential, np.exp(-radii))
        inner = (radii > 1e-2) & (radii < 20.0)
        np.testing.assert_allclose(derivative[inner], -np.exp(-radii[inner]), rtol=1e-8)
        # The Martins assumption c = a would be wrong on this grid.
        wrong = log_grid_derivative(replace(potential, radial_parameter_c=None), np.exp(-radii))
        self.assertGreater(np.max(np.abs(wrong[inner] / derivative[inner] - 1.0)), 1e-3)
        # The local-potential table ends with the Coulomb slope.
        table = local_potential_derivative_table(potential, 2)
        self.assertAlmostEqual(table[-1], 2.0 * 3.0 / radii[-1] ** 2, delta=1e-12)


SPECIES_INPUT = """Boundary_Conditions: cluster
Boundary_Sphere_Radius: 6.0 bohr
Grid_Spacing: 0.6
States_Num: 8
Atom_Types_Num: {count}
{species}
"""


class SpeciesCursorTests(unittest.TestCase):
    """The n-th occurrence of a label goes to the n-th species that reads it."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        for name in ("Na_POTRW.DAT", "Si_FHIPP.DAT", "Mg_FHIPP.DAT", "Si_POTR.DAT"):
            (self.directory / name).write_bytes((DATA / name).read_bytes())
        (self.directory / "H_POTRE.DAT").write_bytes((DATA / "spin_ca_H_POTRE.DAT").read_bytes())

    def tearDown(self) -> None:
        self._directory.cleanup()

    def parse(self, species: str, count: int):
        path = self.directory / "parsec.in"
        path.write_text(SPECIES_INPUT.format(species=species, count=count), encoding="utf-8")
        return parse_parsec_input(path)

    @staticmethod
    def block(symbol: str, lines: str, position: str = "0 0 0") -> str:
        return f"Atom_Type: {symbol}\n{lines}\nbegin Atom_Coord\n{position}\nend Atom_Coord\n"

    def test_martins_wang_reads_no_local_component(self) -> None:
        # tests/triclinic: Na (martins_wang) and a second species with the
        # only Local_Component line.
        translation = self.parse(
            self.block("Na", "Pseudopotential_Format: Martins_Wang\nCore_Cutoff_Radius: 3.1")
            + self.block("H", "Local_Component: s", "0 0 3"),
            2,
        )
        species = translation.problem.pseudopotentials
        self.assertEqual(species["Na"].pseudopotential_format, "martins_wang")
        self.assertEqual(species["Na"].local_angular_momentum, 1)
        self.assertEqual(species["Na"].core_cutoff_radius, 3.1)
        self.assertIsNone(species["Na"].number_of_channels)
        self.assertEqual(species["H"].local_angular_momentum, 0)
        self.assertEqual(species["Na"].path.name, "Na_POTRW.DAT")
        # A Local_Component under the Martins-Wang species would go to H.
        with self.assertRaisesRegex(ParsecInputError, "remove the lines of the martins_wang"):
            self.parse(
                self.block("Na", "Pseudopotential_Format: martins_wang\nLocal_Component: d\nCore_Cutoff_Radius: 3.1")
                + self.block("H", "Local_Component: s", "0 0 3"),
                2,
            )
        with self.assertRaisesRegex(ParsecInputError, "needs Core_Cutoff_Radius"):
            self.parse(self.block("Na", "Pseudopotential_Format: martins_wang"), 1)

    def test_cursors_of_core_radius_orbitals_and_core_correction(self) -> None:
        # Core_Cutoff_Radius is read by old, wang and fhipp; Electron_Per_Orbital
        # by old and fhipp; Nonlinear_Core_Correction by fhipp only.
        species = (
            # Pseudopotential_Format is read for every species: without its
            # own line H would take the first one, fhipp, as in PARSEC.
            self.block(
                "H",
                "Pseudopotential_Format: martins_new\nLocal_Component: s\nPotential_Num: 1",
                "0 0 4",
            )
            + self.block(
                "Mg",
                "Pseudopotential_Format: fhipp\nLocal_Component: d\nPotential_Num: 4\n"
                "Core_Cutoff_Radius: 2.4\nNonlinear_Core_Correction: .true.\n"
                "begin Electron_Per_Orbital\n2 0 0 0\nend Electron_Per_Orbital",
            )
            + self.block("Na", "Pseudopotential_Format: martins_wang\nPotential_Num: 9\nCore_Cutoff_Radius: 3.1", "0 0 -4")
            + self.block(
                "Si",
                "Pseudopotential_Format: martins_old\nLocal_Component: p\nPotential_Num: 2\n"
                "Core_Cutoff_Radius: 2.5\n"
                "begin Electron_Per_Orbital\n2 2 0\nend Electron_Per_Orbital",
                "4 0 0",
            )
        )
        translation = self.parse(species, 4)
        result = translation.problem.pseudopotentials
        self.assertEqual(result["Mg"].core_cutoff_radius, 2.4)
        self.assertTrue(result["Mg"].nonlinear_core_correction)
        self.assertEqual(result["Mg"].electrons_per_orbital, (2.0, 0.0, 0.0, 0.0))
        self.assertEqual(result["Mg"].local_angular_momentum, 2)
        self.assertEqual(result["Na"].core_cutoff_radius, 3.1)
        self.assertEqual(result["Si"].core_cutoff_radius, 2.5)
        self.assertEqual(result["Si"].electrons_per_orbital, (2.0, 2.0))
        self.assertEqual(result["Si"].local_angular_momentum, 1)
        self.assertFalse(result["Si"].nonlinear_core_correction)
        loaded = load_pseudopotentials(result)
        self.assertTrue(loaded["Mg"].has_nonlinear_core_correction)
        self.assertEqual(loaded["Mg"].ionic_charge, 2.0)
        # A second Nonlinear_Core_Correction has no fhipp species to read it.
        with self.assertRaisesRegex(ParsecInputError, "nonlinear_core_correction occurs 2 times"):
            self.parse(
                species.replace("Potential_Num: 2\n", "Potential_Num: 2\nNonlinear_Core_Correction: true\n"),
                4,
            )

    def test_refusals(self) -> None:
        fhipp = (
            "Pseudopotential_Format: fhipp\nLocal_Component: d\nPotential_Num: 3\n"
            "Core_Cutoff_Radius: 2.5\nbegin Electron_Per_Orbital\n2 2 0\nend Electron_Per_Orbital"
        )
        # usrinputfile.F90 error 135.
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "SO_PSP is not supported"):
            self.parse(self.block("Si", fhipp + "\nSO_PSP: .true."), 1)
        # usrinputfile.F90 error 234.
        with self.assertRaisesRegex(ParsecInputError, "a fhipp file holds no valence"):
            self.parse(self.block("Si", fhipp + "\nRead_VCD: .true."), 1)
        with self.assertRaisesRegex(ParsecInputError, "needs an Electron_Per_Orbital block"):
            self.parse(
                self.block("Si", fhipp.replace("begin Electron_Per_Orbital\n2 2 0\nend Electron_Per_Orbital", "")),
                1,
            )
        # usrinputfile.F90 error 134.
        with self.assertRaisesRegex(ParsecInputError, "unknown Pseudopotential_Format"):
            self.parse(self.block("Si", "Pseudopotential_Format: fhi\nLocal_Component: d"), 1)
        translation = self.parse(self.block("Si", fhipp), 1)
        self.assertFalse(translation.problem.pseudopotentials["Si"].nonlinear_core_correction)
        self.assertEqual(translation.problem.pseudopotentials["Si"].path.name, "Si_FHIPP.DAT")


if __name__ == "__main__":
    unittest.main()
