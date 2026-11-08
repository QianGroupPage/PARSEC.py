"""Dipole, finite-field polarizability, point charges and inert keywords.

The features of ``dipole.f90``, ``polar.f90``, ``forcept.f90``/``ionpot.f90``,
``symm_scalar`` (``newrho.F90``) and ``nclmom`` (``bxc.F90``).  Each formula
is checked on small inputs; the text of ``polar.dat`` against a file the
Fortran program wrote for water (Fortran parity round:
H2O, h = 0.3 ang, R = 5.5 ang, Electric_Field 0.005).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest.mock
import unittest

import numpy as np

from parsec_python.Energy.dipole import (
    DEBYE_PER_E_BOHR,
    DipoleMoment,
    cluster_dipole_moment,
    dipole_moment,
)
from parsec_python.Forces import point_charge_forces
from parsec_python.Grid import (
    build_cluster_grid,
    grid_point_group,
    symmetrize_density,
    symmetrize_scalar,
)
from parsec_python.Forces.symmetry import atom_symmetry_operations
from parsec_python.Input import ParsecInputError, parse_parsec_input
from parsec_python.Input.parsec_input import UnsupportedParsecOptionError
from parsec_python.Output.parsec_output import (
    ParsecTextReporter,
    dipole_report_lines,
    point_charge_report_lines,
)
from parsec_python.Output.polarizability_output import (
    polar_header,
    polar_run_line,
    polar_summary,
)
from parsec_python.SCF.noncollinear import atomic_sphere_masks, noncollinear_moments
from parsec_python.SCF.polarizability import (
    FIELD_DIRECTIONS,
    FieldRun,
    PolarizabilityResult,
    run_polarizability,
)
from parsec_python.SCF.single_point import placed_point_charges
from parsec_python.V_ion import (
    cluster_centering_shift,
    electric_field_potential,
    point_charge_ion_energy,
    point_charge_potential,
    shifted_point_charges,
)
from parsec_python.models import Atom, GridSettings, PointCharge


DATA = Path(__file__).resolve().parents[1] / "data"

#: polar.dat of the Fortran program for water (see the module docstring).
FORTRAN_POLAR_DAT = """\
 Tot. number of atoms is:            3
 Number of different types of atoms is:            2

  2 H  atoms with coordinates:
     0.000000    -0.519675     1.795241
     0.000000    -0.519675    -1.795241

  1 O  atoms with coordinates:
     0.000000     0.519675     0.000000

 Applied Electric field  =  0.00500 [Ry/au]

   Field           Tot.              Dipole  Moment [Deb.]
 Component      Energy [Ry]          X          Y          Z
  0. 0. 0.   -37.166208760508     0.000003  -2.610596  -0.000025
  1. 0. 0.   -37.166413475070     0.104059  -2.610526  -0.000000
 -1. 0. 0.   -37.166413475054    -0.104054  -2.610526   0.000000
  0. 1. 0.   -37.135416768158    -0.000017  -2.471389   0.000000
  0.-1. 0.   -37.197543144767    -0.000000  -2.747232  -0.000000
  0. 0. 1.   -37.166513503055    -0.000000  -2.609875   0.154836
  0. 0.-1.   -37.166513503000     0.000000  -2.609860  -0.154819

 Tot. charge of the cluster:   0.000

 Polarizability of the cluster (A**3):
                 alfaXX    alfaYY    alfaZZ       average
 From energy:     1.2134    1.6075    1.8063       1.5424
 From dipole:     1.2141    1.6093    1.8065       1.5433

 Polarizability per atom =    0.5141 A**3  (energy)
                              0.5144 A**3  (dipole)
"""

HYDROGEN = SimpleNamespace(ionic_charge=1.0)
OXYGEN = SimpleNamespace(ionic_charge=6.0)


@dataclass(frozen=True)
class _FakeSystem:
    """What the polarizability loop reads of a prepared system."""

    grid: object
    ionic_potential: np.ndarray
    atoms: tuple
    pseudopotentials: dict
    input: object


def _write_input(directory: Path, extra: str = "", atoms: str | None = None) -> Path:
    """A two-atom H2 input on a coarse grid, with ``extra`` lines."""

    text = f"""\
Boundary_Sphere_Radius: 4.0 bohr
Grid_Spacing: 0.8 bohr
Coordinate_Unit: Cartesian_Bohr
Expansion_Order: 4
States_Num: 3
Max_Iter: 3
Eigensolver: chebff
Hartree_Boundary_Tolerance: off
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
{atoms or "  0.0  0.0  0.7\n  0.0  0.0 -0.7"}
end Atom_Coord
{extra}
"""
    path = directory / "parsec.in"
    path.write_text(text, encoding="utf-8")
    (directory / "H_POTRE.DAT").write_bytes((DATA / "water_H_POTRE.DAT").read_bytes())
    return path


class DipoleTests(unittest.TestCase):
    def test_sums_of_dipole_f90(self) -> None:
        rng = np.random.default_rng(3)
        coordinates = rng.normal(size=(50, 3))
        density = rng.random(50)
        positions = np.array([[0.1, 0.2, 0.3], [-0.5, 0.4, 0.0]])
        charges = np.array([1.0, 6.0])
        moment = dipole_moment(density, coordinates, 0.125, positions, charges)
        self.assertAlmostEqual(moment.charge, 7.0 - 0.125 * density.sum(), places=13)
        np.testing.assert_allclose(
            moment.dipole,
            -0.125 * (density[:, None] * coordinates).sum(axis=0) + charges @ positions,
            rtol=0,
            atol=1e-13,
        )
        np.testing.assert_allclose(moment.dipole_debye, 2.54 * moment.dipole)
        self.assertEqual(DEBYE_PER_E_BOHR, 2.54)

    def test_displaced_gaussian_against_its_centre(self) -> None:
        grid = build_cluster_grid(GridSettings(spacing=0.25, radius=6.0))
        centre = np.array([0.3, -0.2, 0.45])
        width = 0.8
        distance = np.sum((grid.coordinates - centre) ** 2, axis=1)
        density = 2.0 * np.exp(-distance / width**2) / (np.pi**1.5 * width**3)
        atoms = (Atom("H", (0.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, 0.0)))
        moment = cluster_dipole_moment(density, grid, atoms, {"H": HYDROGEN})
        # Two electrons around the centre and two protons at the origin.
        self.assertAlmostEqual(moment.charge, 0.0, places=8)
        np.testing.assert_allclose(moment.dipole, -2.0 * centre, atol=1e-8)

    def test_lines_of_dipole_f90(self) -> None:
        moment = DipoleMoment(charge=-1.0e-12, dipole=np.array([0.0, -2.61057528, 0.0]) / 2.54)
        self.assertEqual(
            dipole_report_lines(moment, 0.0),
            [
                " Tot. charge (user input) [e] =   0.00000000",
                " Tot. charge (computed) [e] =  -0.00000000",
                " Dipole components [Debye]:",
                " x-component =   0.00000000",
                " y-component =  -2.61057528",
                " z-component =   0.00000000",
            ],
        )


class PointChargeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.charges = (
            PointCharge(0.5, (0.3, 2.6, 0.4)),
            PointCharge(-0.3, (-2.2, -1.5, 0.7)),
        )
        self.atoms = (
            Atom("O", (0.0, 0.1, 0.0)),
            Atom("H", (0.0, -0.5, 1.8)),
            Atom("H", (0.0, -0.5, -1.8)),
        )
        self.potentials = {"O": OXYGEN, "H": HYDROGEN}

    def test_potential_of_ionpot_f90(self) -> None:
        points = np.array(
            [[0.3, 2.6, 0.4 + 0.5], [0.3, 2.6, 0.4 + 1.0], [0.3, 2.6, 0.4 + 2.0], [5.0, 5.0, 5.0]]
        )
        potential = point_charge_potential(points, self.charges[:1])
        # Within one bohr (and at exactly one) nothing; outside -2q/r.
        np.testing.assert_allclose(potential, [0.0, 0.0, -2.0 * 0.5 / 2.0, -1.0 / np.linalg.norm(points[3] - (0.3, 2.6, 0.4))])
        both = point_charge_potential(points, self.charges)
        np.testing.assert_allclose(
            both,
            potential + point_charge_potential(points, self.charges[1:]),
        )

    def test_force_is_minus_the_gradient_of_the_energy(self) -> None:
        forces = point_charge_forces(self.atoms, self.potentials, self.charges)
        step = 1.0e-5
        for index in range(len(self.atoms)):
            for axis in range(3):
                shifted = []
                for sign in (1.0, -1.0):
                    moved = list(self.atoms)
                    position = np.array(moved[index].position)
                    position[axis] += sign * step
                    moved[index] = Atom(moved[index].symbol, position)
                    shifted.append(point_charge_ion_energy(moved, self.potentials, self.charges))
                derivative = (shifted[0] - shifted[1]) / (2.0 * step)
                self.assertAlmostEqual(forces[index, axis], -derivative, places=7)

    def test_energy_of_forcept_f90(self) -> None:
        expected = sum(
            2.0 * charge.charge * self.potentials[atom.symbol].ionic_charge
            / np.linalg.norm(np.asarray(atom.position) - charge.position)
            for charge in self.charges
            for atom in self.atoms
        )
        self.assertAlmostEqual(
            point_charge_ion_energy(self.atoms, self.potentials, self.charges), expected, places=13
        )
        self.assertEqual(point_charge_ion_energy(self.atoms, self.potentials, ()), 0.0)

    def test_charges_move_with_the_atoms(self) -> None:
        shift = cluster_centering_shift(self.atoms)
        np.testing.assert_allclose(shift, [0.0, -0.2, 0.0])
        moved = shifted_point_charges(self.charges, shift)
        np.testing.assert_allclose(moved[0].position, [0.3, 2.8, 0.4])
        problem = SimpleNamespace(atoms=self.atoms, point_charges=self.charges, recenter_geometry=True)
        np.testing.assert_allclose(placed_point_charges(problem)[1].position, [-2.2, -1.3, 0.7])
        kept = SimpleNamespace(atoms=self.atoms, point_charges=self.charges, recenter_geometry=False)
        self.assertIs(placed_point_charges(kept)[0], self.charges[0])
        small = (Atom("H", (0.0, 0.0, 0.05)), Atom("H", (0.0, 0.0, -0.01)))
        self.assertIsNone(cluster_centering_shift(small))

    def test_lines_of_usrinputfile_and_initial(self) -> None:
        placed = shifted_point_charges(self.charges, (0.0, -0.403, 0.0))
        lines = point_charge_report_lines(self.charges, placed, (1, 1))
        self.assertEqual(lines[:7], [
            " Point Charge data:",
            " ------------------",
            " Tot. # of point charge types is            2",
            " There are    1 point charges with charge  0.50 e",
            " and their coordinates are:",
            "",
            "    x [bohr]          y [bohr]          z [bohr]",
        ])
        self.assertEqual(lines[7], "    0.300000000       2.600000000       0.400000000")
        self.assertIn(" Tot. number of point charges =    2", lines)
        self.assertIn("WARNING: point charge coordinates were shifted", lines)
        self.assertEqual(lines[-2], "   -2.200000000      -1.097000000       0.700000000")
        self.assertNotIn(
            "WARNING: point charge coordinates were shifted",
            point_charge_report_lines(self.charges, self.charges, (2,)),
        )


class ElectricFieldTests(unittest.TestCase):
    def test_potential_of_polar_f90(self) -> None:
        points = np.array([[1.0, 2.0, 3.0], [-0.5, 0.0, 4.0]])
        np.testing.assert_allclose(
            electric_field_potential(points, 0.005, (0.0, -1.0, 0.0)), [-0.02, 0.0]
        )
        np.testing.assert_allclose(
            electric_field_potential(points, 0.005, (1.0, 0.0, 0.0)), [0.01, -0.005]
        )

    def test_units_of_esdf(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for text, expected in (
                ("", 0.001),
                ("Electric_Field: 0.004", 0.004),
                ("Electric_Field: 0.002 ry/bohr/e", 0.002),
                ("Electric_Field: 0.002 Har/Bohr/e", 0.004),
                ("Electric_Field: 2.5711273e9 V/m", 0.01),
                ("Electric_Field: 2.5711273 v/nm", 0.01),
                ("Electric_Field: 0.25711273 v/ang", 0.01),
                ("Electric_Field: 1.8897268 v/bohr", 1.8897268 * 1.8897268e10 / 2.5711273e11),
            ):
                path = _write_input(directory, "Polarizability: .true.\n" + text)
                translation = parse_parsec_input(path)
                self.assertAlmostEqual(translation.polarizability_field, expected, places=14, msg=text)
            path = _write_input(directory, "Polarizability: .true.\nElectric_Field: 1 kV/m")
            with self.assertRaises(UnsupportedParsecOptionError):
                parse_parsec_input(path)


class PolarizabilityTests(unittest.TestCase):
    def fortran_result(self) -> PolarizabilityResult:
        lines = FORTRAN_POLAR_DAT.splitlines()[14:21]
        runs = []
        for line, direction in zip(lines, FIELD_DIRECTIONS):
            values = [float(value) for value in line[10:].split()]
            runs.append(
                FieldRun(
                    direction=direction,
                    total_energy=values[0],
                    dipole=DipoleMoment(charge=0.0, dipole=np.array(values[1:]) / 2.54),
                    converged=True,
                    iterations=10,
                )
            )
        return PolarizabilityResult(field=0.005, runs=tuple(runs), atom_count=3)

    def test_polar_dat_of_the_fortran_run(self) -> None:
        atoms = (
            Atom("H", (0.0, -0.519674891, 1.795240534)),
            Atom("H", (0.0, -0.519674891, -1.795240534)),
            Atom("O", (0.0, 0.519674891, 0.0)),
        )
        result = self.fortran_result()
        text = polar_header(atoms, ["H", "O"], 0.005)
        for run in result.runs:
            text += polar_run_line(run.direction, run.total_energy, run.dipole.dipole_debye)
        text += polar_summary(result)
        self.assertEqual(text, FORTRAN_POLAR_DAT)

    def test_formulas_on_a_quadratic_model(self) -> None:
        alpha = np.array([7.0, 9.0, 11.0])
        mu0 = np.array([0.1, -0.3, 0.2])
        field = 0.004
        runs = []
        for direction in FIELD_DIRECTIONS:
            vector = field * np.asarray(direction)
            energy_hartree = -10.0 - mu0 @ vector - 0.5 * alpha @ vector**2
            runs.append(
                FieldRun(direction, 2.0 * energy_hartree, DipoleMoment(0.0, mu0 + alpha * vector), True, 1)
            )
        result = PolarizabilityResult(field=field, runs=tuple(runs), atom_count=2)
        cubic = 0.529177**3
        np.testing.assert_allclose(result.from_energy, alpha * cubic, rtol=1e-9)
        np.testing.assert_allclose(result.from_dipole, alpha * cubic, rtol=1e-12)

    def test_loop_of_parsec_f90(self) -> None:
        """Seven runs, the field of each in V_ion and pot%vnew = pot%vold."""

        grid = build_cluster_grid(GridSettings(spacing=1.0, radius=3.0))
        base = np.linspace(-1.0, 1.0, grid.size)
        system = _FakeSystem(
            grid=grid,
            ionic_potential=base,
            atoms=(Atom("H", (0.0, 0.0, 0.0)),),
            pseudopotentials={"H": HYDROGEN},
            input=SimpleNamespace(periodic_cell=None),
        )
        calls = []

        def scf_runner(run_system, *, callback=None, warm_start=None):
            index = len(calls)
            calls.append((run_system, warm_start))
            return SimpleNamespace(
                energies=SimpleNamespace(total=-1.0 - 0.01 * index),
                density=np.full(grid.size, 1.0 / (grid.size * grid.volume_element)),
                converged=True,
                iterations=3,
                input_effective_potential=np.full(grid.size, float(index)),
                hartree_potential=np.full(grid.size, 10.0 + index),
                eigensolver_state=None,
                spin=None,
            )

        result, last_system, _ = run_polarizability(system, 0.005, scf_runner=scf_runner)
        self.assertEqual(len(calls), 7)
        self.assertIsNone(calls[0][1])
        for index, ((run_system, warm), direction) in enumerate(zip(calls, FIELD_DIRECTIONS)):
            np.testing.assert_allclose(
                run_system.ionic_potential,
                base + 2.0 * 0.005 * (grid.coordinates @ np.asarray(direction)),
            )
            if index:
                # The input potential of the last iteration of the run before.
                np.testing.assert_array_equal(warm.input_potential, np.full(grid.size, index - 1.0))
                np.testing.assert_array_equal(warm.hartree_potential, np.full(grid.size, 9.0 + index))
        self.assertIs(last_system, calls[-1][0])
        self.assertEqual([run.direction for run in result.runs], list(FIELD_DIRECTIONS))
        self.assertAlmostEqual(result.charge, 0.0, places=12)


class DensitySymmetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = build_cluster_grid(GridSettings(spacing=0.7, radius=3.5))
        atoms = (
            Atom("O", (0.0, 0.0, 0.3)),
            Atom("H", (0.0, 1.4, -0.4)),
            Atom("H", (0.0, -1.4, -0.4)),
        )
        operations = atom_symmetry_operations(atoms)
        self.assertEqual(len(operations), 4)  # C2v
        self.group = grid_point_group(self.grid, np.array([item.matrix for item in operations]))

    def test_average_is_a_symmetric_projection(self) -> None:
        values = np.random.default_rng(5).random(self.grid.size)
        symmetric, report = symmetrize_density(values, self.group, self.grid.volume_element)
        np.testing.assert_allclose(symmetrize_scalar(symmetric, self.group), symmetric, atol=1e-15)
        for images in self.group.images:
            np.testing.assert_allclose(symmetric[images], symmetric, atol=1e-15)
        self.assertAlmostEqual(report.charge_after, report.charge_before, places=12)
        self.assertAlmostEqual(
            report.maximum_deviation_percent,
            # C2v is Abelian with diagonal matrices: PARSEC's wedge holds a
            # quarter of the grid.
            np.max(np.abs(symmetric - values)) / (np.sum(values) / 4.0) * 100.0,
        )
        _, again = symmetrize_density(symmetric, self.group, self.grid.volume_element)
        self.assertLess(again.maximum_deviation_percent, 1e-12)

    def test_images_are_permutations(self) -> None:
        for images in self.group.images:
            self.assertEqual(np.unique(images).size, self.grid.size)
        np.testing.assert_array_equal(self.group.images[0], np.arange(self.grid.size))


class NoncollinearMomentTests(unittest.TestCase):
    def test_moments_of_nclmom(self) -> None:
        coordinates = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        atoms = (Atom("Fe", (0.0, 0.0, 0.0)), Atom("Fe", (3.0, 0.0, 0.0)))
        magnetization = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, -4.0]]
        )
        masks = atomic_sphere_masks(coordinates, atoms, [0.6, 0.6])
        moments = noncollinear_moments(magnetization, masks, 0.5)
        np.testing.assert_allclose(moments.atomic, [[0.5, 1.0, 0.0], [0.0, 0.0, -2.0]])
        np.testing.assert_allclose(moments.total, [0.5, 1.0, -0.5])
        self.assertAlmostEqual(moments.absolute, 0.5 * (1.0 + 2.0 + 3.0 + 4.0))
        # A periodic image within the radius counts the point (nonloc.F90).
        periodic = atomic_sphere_masks(coordinates, atoms[:1], [1.1], np.diag([4.0, 9.0, 9.0]))
        np.testing.assert_array_equal(periodic[0], [True, True, False, True])


class InputTests(unittest.TestCase):
    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.addCleanup(self._folder.cleanup)
        self.directory = Path(self._folder.name)

    def parse(self, extra: str = "", atoms: str | None = None):
        return parse_parsec_input(_write_input(self.directory, extra, atoms))

    def test_defaults_change_nothing(self) -> None:
        translation = self.parse()
        self.assertIsNone(translation.polarizability_field)
        self.assertEqual(translation.problem.point_charges, ())
        self.assertFalse(translation.problem.scf.symmetrize_density)
        self.assertNotIn("point_charges", repr(translation.problem))
        self.assertNotIn("symmetrize_density", repr(translation.problem.scf))
        self.assertNotIn("atomic_radius", repr(translation.problem.pseudopotentials["H"]))
        self.assertEqual(translation.warnings, ())

    def test_polarizability(self) -> None:
        translation = self.parse("Polarizability: .true.")
        self.assertEqual(translation.polarizability_field, 0.001)
        self.assertFalse(translation.use_symmetry)
        with self.assertRaisesRegex(ParsecInputError, "polarizability and move atoms"):
            self.parse("Polarizability: .true.\nMinimization: bfgs")
        with self.assertRaisesRegex(ParsecInputError, "polarizability and move atoms"):
            self.parse("Polarizability: .true.\nMolecular_Dynamics: .true.\nTinit: 300 K")
        ignored = self.parse("Electric_Field: 0.01")
        self.assertIsNone(ignored.polarizability_field)
        self.assertTrue(any("Electric_Field is ignored" in text for text in ignored.warnings))

    def test_point_charges(self) -> None:
        translation = self.parse(
            "Add_Point_Charges: .true.\nPoint_Typ_Num: 2\n"
            "Pt_Chg: 0.5\nbegin Point_Coord\n 1.0 2.0 3.0\n 0.0 0.0 3.0\nend Point_Coord\n"
            "Pt_Chg: -1 e\nbegin Point_Coord\n -3.0 0.0 0.0\nend Point_Coord"
        )
        charges = translation.problem.point_charges
        self.assertEqual(translation.point_charge_type_counts, (2, 1))
        self.assertEqual([item.charge for item in charges], [0.5, 0.5, -1.0])
        np.testing.assert_allclose(charges[2].position, [-3.0, 0.0, 0.0])
        self.assertFalse(translation.use_symmetry)
        self.assertIn("point_charges=", repr(translation.problem))
        with self.assertRaisesRegex(ParsecInputError, "Point_Typ_Num"):
            self.parse("Add_Point_Charges: .true.\nbegin Point_Coord\n 1 2 3\nend Point_Coord")
        with self.assertRaisesRegex(ParsecInputError, "no point charge coordinates"):
            self.parse("Add_Point_Charges: .true.\nPoint_Typ_Num: 1\nPt_Chg: 1")
        ignored = self.parse("Pt_Chg: 1\nbegin Point_Coord\n 1 2 3\nend Point_Coord")
        self.assertEqual(ignored.problem.point_charges, ())
        self.assertTrue(any("without Add_Point_Charges" in text for text in ignored.warnings))

    def test_point_charges_in_angstrom(self) -> None:
        text = _write_input(self.directory).read_text().replace(
            "Coordinate_Unit: Cartesian_Bohr", "Coordinate_Unit: Cartesian_Ang"
        )
        text += "Add_Point_Charges: .true.\nPoint_Typ_Num: 1\nbegin Point_Coord\n 0.529177 0 0\nend Point_Coord\n"
        (self.directory / "parsec.in").write_text(text)
        charges = parse_parsec_input(self.directory / "parsec.in").problem.point_charges
        # A missing Pt_Chg is zero, as esdf_physical's default.
        self.assertEqual(charges[0].charge, 0.0)
        np.testing.assert_allclose(charges[0].position, [1.0, 0.0, 0.0], atol=1e-12)

    def test_inert_keywords(self) -> None:
        translation = self.parse(
            "External_Mag_Field: 5.0e9\nOrb_Mag: .true.\nOld_Pseudopotential_Format: .true."
        )
        self.assertAlmostEqual(translation.magnetic_field_gauss, 2.0)
        self.assertTrue(translation.complex_wavefunctions)
        warnings = " ".join(translation.warnings)
        for word in ("External_Mag_Field", "Orb_Mag", "Old_Pseudopotential_Format", "Complex"):
            self.assertIn(word, warnings)
        self.assertTrue(self.parse("Complex_Wfn: .true.").complex_wavefunctions)
        self.assertFalse(self.parse("Complex_Wfn: .false.").complex_wavefunctions)

    def test_old_pseudopotential_format_with_an_old_species(self) -> None:
        (self.directory / "Si_POTR.DAT").write_bytes((DATA / "Si_POTR.DAT").read_bytes())
        text = """\
Old_Pseudopotential_Format: .true.
Boundary_Sphere_Radius: 4.0 bohr
Grid_Spacing: 0.8 bohr
States_Num: 4
Atom_Types_Num: 1
Atom_Type: Si
Pseudopotential_Format: martins_old
Core_Cutoff_Radius: 2.5
Local_Component: p
Potential_Num: 2
begin Electron_Per_Orbital
2 2 0
end Electron_Per_Orbital
begin Atom_Coord
 0.0 0.0 0.0
end Atom_Coord
"""
        (self.directory / "parsec.in").write_text(text)
        translation = parse_parsec_input(self.directory / "parsec.in")
        self.assertEqual(
            translation.problem.pseudopotentials["Si"].pseudopotential_format, "martins_old"
        )
        self.assertTrue(any("Old_Pseudopotential_Format" in text for text in translation.warnings))

    def test_symmetrize_charge_density(self) -> None:
        translation = self.parse("Symmetrize_Charge_Density: .true.")
        self.assertTrue(translation.problem.scf.symmetrize_density)
        self.assertIn("symmetrize_density=True", repr(translation.problem.scf))
        off = self.parse("Symmetrize_Charge_Density: .true.\nIgnore_Symmetry: .true.")
        self.assertFalse(off.problem.scf.symmetrize_density)
        self.assertTrue(any("Symmetrize_Charge_Density changes nothing" in t for t in off.warnings))

    def test_atomic_radius(self) -> None:
        translation = self.parse("Atomic_Radius: 1.5")
        self.assertEqual(translation.problem.pseudopotentials["H"].atomic_radius, 1.5)
        self.assertTrue(any("Atomic_Radius" in text for text in translation.warnings))

    def test_header_lines(self) -> None:
        translation = self.parse(
            "Polarizability: .true.\nElectric_Field: 0.005\nExternal_Mag_Field: 2.5e9\n"
            "Symmetrize_Charge_Density: .true."
        )
        written: list[str] = []
        ParsecTextReporter(written.append, translation).header()
        text = written[0].splitlines()
        self.assertIn(" WAVEFUNCTIONS ARE COMPLEX!", text)
        self.assertIn(" Magnetic field: H =    1.00000 Gauss", text)
        self.assertIn(" Applying external uniform magnetic field", text)
        self.assertEqual(
            text[text.index(" Polarizability data:") : text.index(" Polarizability data:") + 4],
            [
                " Polarizability data:",
                " --------------------",
                " Using seven-fields polarizability approach",
                " Electric field  =  0.00500 [Ry/au]",
            ],
        )
        # Polarizability switches the point group off, so no symmetrization.
        self.assertNotIn(
            " Symmetrizing charge density according to symm. operations in the full point group",
            text,
        )
        plain: list[str] = []
        ParsecTextReporter(plain.append, self.parse()).header()
        self.assertIn(" WAVEFUNCTIONS ARE REAL!", plain[0].splitlines())
        self.assertNotIn("Polarizability", plain[0])


class SinglePointTests(unittest.TestCase):
    """The point charges and the symmetrized density in a prepared system."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.addCleanup(self._folder.cleanup)
        self.directory = Path(self._folder.name)

    def test_point_charges_in_the_system(self) -> None:
        from parsec_python.Forces import compute_forces
        from parsec_python.SCF import prepare_single_point, run_scf, update_geometry

        extra = (
            "Add_Point_Charges: .true.\nPoint_Typ_Num: 1\nPt_Chg: 0.4\n"
            "begin Point_Coord\n 0.5 1.5 0.2\nend Point_Coord"
        )
        atoms = "  0.0  0.4  0.7\n  0.0  0.4 -0.7"
        charged = parse_parsec_input(_write_input(self.directory, extra, atoms)).problem
        plain = replace(charged, point_charges=())
        system = prepare_single_point(charged)
        reference = prepare_single_point(plain)
        # The cluster is recentred by (0, 0.4, 0), and the charge with it.
        np.testing.assert_allclose(system.point_charges[0].position, [0.5, 1.1, 0.2])
        np.testing.assert_allclose(
            system.ionic_potential - reference.ionic_potential,
            point_charge_potential(system.grid.coordinates, system.point_charges),
            atol=1e-13,
        )
        self.assertAlmostEqual(
            system.ion_ion_energy - reference.ion_ion_energy,
            point_charge_ion_energy(system.atoms, system.pseudopotentials, system.point_charges),
            places=12,
        )
        moved_atoms = (Atom("H", (0.0, 0.1, 0.72)), Atom("H", (0.0, 0.0, -0.7)))
        moved = update_geometry(system, moved_atoms)
        moved_reference = update_geometry(reference, moved_atoms)
        np.testing.assert_allclose(
            moved.ionic_potential - moved_reference.ionic_potential,
            point_charge_potential(system.grid.coordinates, system.point_charges),
            atol=1e-13,
        )
        self.assertAlmostEqual(
            moved.ion_ion_energy - moved_reference.ion_ion_energy,
            point_charge_ion_energy(moved_atoms, system.pseudopotentials, system.point_charges),
            places=12,
        )
        result = run_scf(system)
        forces = compute_forces(system, result, symmetrize=False)
        np.testing.assert_allclose(
            forces.point_charge,
            point_charge_forces(system.atoms, system.pseudopotentials, system.point_charges),
        )
        self.assertIsNone(compute_forces(reference, run_scf(reference), symmetrize=False).point_charge)

    def test_symmetrized_density_in_the_loop(self) -> None:
        from parsec_python.SCF import prepare_single_point, run_scf

        translation = parse_parsec_input(
            _write_input(self.directory, "Symmetrize_Charge_Density: .true.")
        )
        system = prepare_single_point(translation.problem)
        self.assertEqual(system.density_symmetry.order, 16)  # D4h of H2 on the z axis
        result = run_scf(system)
        for images in system.density_symmetry.images:
            np.testing.assert_allclose(result.density[images], result.density, atol=1e-15)
        report = result.history[0].density_symmetrization[0]
        self.assertAlmostEqual(report.charge_after, report.charge_before, places=10)
        plain = prepare_single_point(replace(translation.problem, scf=replace(translation.problem.scf, symmetrize_density=False)))
        self.assertIsNone(plain.density_symmetry)
        self.assertEqual(run_scf(plain).history[0].density_symmetrization, ())


    def test_spin_polarized_field_runs_and_symmetrized_channels(self) -> None:
        from parsec_python.SCF import prepare_single_point, run_scf

        spin = "Spin_Polarization: .true.\nCorrelation_Type: ca\n"
        problem = parse_parsec_input(
            _write_input(self.directory, spin + "Polarizability: .true.\nElectric_Field: 0.01")
        ).problem
        result, _, last = run_polarizability(prepare_single_point(problem), 0.01)
        self.assertEqual(len(result.runs), 7)
        self.assertIsNotNone(last.spin)
        # The field along +z pulls the electrons to -z: a dipole along +z.
        self.assertGreater(result.runs[5].dipole.dipole[2], result.runs[6].dipole.dipole[2])
        symmetric = parse_parsec_input(
            _write_input(self.directory, spin + "Symmetrize_Charge_Density: .true.")
        ).problem
        spin_result = run_scf(prepare_single_point(symmetric))
        reports = spin_result.history[-1].density_symmetrization
        self.assertEqual(len(reports), 3)
        self.assertAlmostEqual(
            reports[0].charge_after, reports[1].charge_after + reports[2].charge_after, places=10
        )


class CommandLineTests(unittest.TestCase):
    def test_reference_run_reports_dipole_and_charges(self) -> None:
        from parsec_python.cli import main

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = _write_input(
                directory,
                "Add_Point_Charges: .true.\nPoint_Typ_Num: 1\nPt_Chg: 0.3\n"
                "begin Point_Coord\n 0.0 1.5 0.0\nend Point_Coord",
            )
            self.assertIn(main([str(path), "--quiet"]), (0, 3))
            text = (directory / "parsec.out").read_text(encoding="utf-8")
            self.assertIn(" Dipole components [Debye]:", text)
            self.assertIn(" Point Charge data:", text)
            self.assertIn("Forces and MPI statistics", text)
            self.assertNotIn("dipoles", text.split("Time for self-consistent field")[-1])
            with np.load(directory / "parsec_python_results.npz") as archive:
                self.assertEqual(archive["dipole_debye"].shape, (3,))
                self.assertAlmostEqual(float(archive["dipole_charge_e"]), 0.0, places=3)

    def test_noncollinear_run_prints_the_atomic_moments(self) -> None:
        from parsec_python.cli import main

        source = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "Au_POTRE.DAT").write_bytes(source.read_bytes())
            (directory / "parsec.in").write_text(
                "Boundary_Sphere_Radius: 6.0\nGrid_Spacing: 0.6\nExpansion_Order: 6\n"
                "States_Num: 20\nFermi_Temp: 500.0\nMax_Iter: 2\n"
                "Eigensolver: chebff\nNon_Collinear_magnetism: .true.\n"
                "Atom_Types_Num: 1\nAtom_Type: Au\nLocal_Component: s\n"
                "Initial_Spin_Polarization: 0.3\nAtomic_Radius: 2.5\n"
                "begin Atom_Coord\n 0.0 0.0 0.0\nend Atom_Coord\n"
                "begin Initial_NCL_Moment\n0 3 4\nend Initial_NCL_Moment\n"
                "Correlation_Type: CA\n",
                encoding="utf-8",
            )
            # The k-point machinery of the spinor drivers runs on the host
            # here: its device converter takes Bloch Hamiltonians only.
            with unittest.mock.patch.dict("os.environ", {"PARSEC_KPOINT_BACKEND": "cpu"}):
                main([str(directory / "parsec.in"), "--quiet"])
            lines = (directory / "parsec.out").read_text(encoding="utf-8").splitlines()
        start = lines.index("Atomic magnetic moments (Bohr magnetons): ")
        self.assertEqual(
            lines[start + 2], "-atom-  --Mx--  --My--  --Mz--   [bohr magnetons]       "
        )
        moment = [float(value) for value in lines[start + 4].split()[1:]]
        # The moment starts along (0, 3, 4) and stays near it.
        self.assertLess(abs(moment[0]), 0.1)
        self.assertGreater(moment[1], 0.0)
        self.assertGreater(moment[2], moment[1])
        self.assertTrue(lines[start + 7].startswith(" Tot. Magnetization (Bohr magnetons)"))
        self.assertTrue(lines[start + 9].startswith(" Integral{|m|} = "))
        self.assertIn(" Dipole components [Debye]:", lines)

    def test_accelerated_cli_runs_the_cluster_features(self) -> None:
        # The accelerated SCF has run them since the GPU phase of the parity
        # round (acceleration/tests/test_gpu_fields.py); only their spin-orbit
        # combinations are handed to the reference solver.
        from parsec_python.acceleration.cli import _reference_only_features

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for extra in (
                "Polarizability: .true.",
                "Add_Point_Charges: .true.\nPoint_Typ_Num: 1\nbegin Point_Coord\n 0 2 0\nend Point_Coord",
                "Symmetrize_Charge_Density: .true.",
            ):
                translation = parse_parsec_input(_write_input(directory, extra))
                features = _reference_only_features(translation, accelerated_backend_selected=False)
                self.assertEqual(features, [])
            self.assertEqual(
                _reference_only_features(
                    parse_parsec_input(_write_input(directory)), accelerated_backend_selected=False
                ),
                [],
            )


if __name__ == "__main__":
    unittest.main()
