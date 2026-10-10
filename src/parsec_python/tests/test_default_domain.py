"""Default of the sphere radius and of the Hartree boundary tolerance.

Three sets of stored values are compared with:

* ``data/parsed_inputs_given_radius.json``: what the parser made of every
  input with a radius before the default rule existed
  (:mod:`.parsed_inputs`);
* ``data/domain_rule_reference.json``: what a reference script of the rule
  (an implementation of its own, not in this tree) gave for every
  shipped input without its radius line and for the constructed ones;
* ``data/domain_measured_shells.json``: the shell sums of 83 measured runs
  on A100, rebinned to 0.4 bohr, with the fit of a reference script
  (not in this tree either) and the measured energy above the widest sphere.

``PARSEC_TEST_DOMAIN_INPUTS`` names a folder of measurement inputs that is
not part of the repository: ``recipe_inputs`` (the benchmark clusters k06 to
k14, C795H300 to C9449H1572) and ``general_inputs`` (ten systems without
hydrogen at the surface); the tests on them are skipped without it.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
from decimal import Decimal
from io import StringIO
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from parsec_python.Grid import build_cluster_grid
from parsec_python.Hartree import (
    estimate_omitted_potential,
    plan_hartree_boundary,
    valence_point_charges,
)
from parsec_python.Hartree import boundary as boundary_module
from parsec_python.Hartree import domain
from parsec_python.Input.parsec_input import (
    ANGSTROM_TO_BOHR,
    ParsecInputError,
    _energy_rydberg,
    _physical_length,
    parse_parsec_input,
)
from parsec_python.Output import ParsecTextReporter
from parsec_python.Output import parsec_output
from parsec_python.Output.parsec_output import (
    domain_finish_lines,
    domain_header_lines,
    domain_report_requested,
    domain_setup_lines,
)
from parsec_python.Pseudopotential import (
    parsec_radial_integral,
    read_parsec_pseudopotential,
)
from parsec_python.SCF.single_point import prepare_single_point
from parsec_python.V_ion import (
    center_cluster_geometry,
    ionic_charge,
    load_pseudopotentials,
)
from parsec_python.cli import main as cli_main
from parsec_python.models import (
    Atom,
    GridSettings,
    HartreeSettings,
    SpeciesPotential,
)

from parsec_python.tests import parsed_inputs
from parsec_python.tests.parsed_inputs import BASE_PSEUDOPOTENTIALS, NANODIAMOND, REPOSITORY


DATA = Path(__file__).parent / "data"
GOLDEN = json.loads((DATA / "parsed_inputs_given_radius.json").read_text(encoding="utf-8"))
REFERENCE = json.loads(
    (DATA / "domain_rule_reference.json").read_text(encoding="utf-8")
)
SHELLS = json.loads((DATA / "domain_measured_shells.json").read_text(encoding="utf-8"))
MEASUREMENTS = os.environ.get("PARSEC_TEST_DOMAIN_INPUTS", "").strip()
needs_measurements = unittest.skipUnless(
    MEASUREMENTS, "PARSEC_TEST_DOMAIN_INPUTS does not name the measurement inputs"
)
# Radius, Hartree boundary tolerance and multipole order of the rule for the
# benchmark clusters without their radius line.  These ran on A100.
BENCHMARK = {
    "k06": ("16.8 ang", "1.0e-03 Ry", 19),
    "k07": ("19.0 ang", "1.0e-03 Ry", 24),
    "k08": ("20.3 ang", "9.4e-04 Ry", 23),
    "k09": ("23.8 ang", "7.8e-04 Ry", 29),
    "k10": ("26.2 ang", "6.6e-04 Ry", 36),
    "k11": ("27.4 ang", "5.7e-04 Ry", 37),
    "k12": ("29.7 ang", "5.1e-04 Ry", 44),
    "k13": ("30.9 ang", "4.6e-04 Ry", 43),
    "k14": ("34.5 ang", "4.0e-04 Ry", 50),
}
BENCHMARK_ELECTRONS = {
    "k06": 3480,
    "k07": 5264,
    "k08": 7120,
    "k09": 10456,
    "k10": 14680,
    "k11": 19392,
    "k12": 23768,
    "k13": 29576,
    "k14": 39368,
}
# Radius, vacuum (ang), wall estimate and its shares for the ten systems
# without hydrogen at the surface: the reference values of the rule.
GENERAL = {
    "cf4": ("4.9 ang", 3.58, "4.4e-04", "C 54 %, F 46 %"),
    "co2": ("4.9 ang", 3.74, "4.3e-04", "C 55 %, O 45 %"),
    "n2": ("4.6 ang", 4.05, "4.4e-04", "N 100 %"),
    "ccl4": ("5.9 ang", 4.13, "4.5e-04", "C 5 %, Cl 95 %"),
    "sif4": ("5.7 ang", 4.15, "4.3e-04", "F 6 %, Si 94 %"),
    "c35_bare": ("8.2 ang", 4.63, "4.4e-04", "C 100 %"),
    "mg2": ("7.2 ang", 5.25, "4.5e-04", "Mg 100 %"),
    "nacl": ("7.3 ang", 6.12, "4.8e-04", "Na 100 %"),
    "na2": ("8.0 ang", 6.46, "4.4e-04", "Na 100 %"),
    "cl_anion": ("7.1 ang", 7.10, None, None),
}


@contextmanager
def input_folder(text: str):
    """A folder with ``parsec.in`` of this text; the path of the file."""

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder).resolve() / "parsec.in"
        path.write_bytes(text.encode("utf-8"))
        yield path


def parse_text(text: str, pseudopotentials: Path):
    with input_folder(text) as path:
        return parse_parsec_input(path, pseudopotential_directory=pseudopotentials)


def source_of(path: Path, pseudopotentials: Path | None) -> tuple[str, Path]:
    """Text of a shipped input and the folder its pseudopotentials come from."""

    return (
        path.read_text(encoding="utf-8").replace("\r\n", "\n"),
        path.parent if pseudopotentials is None else pseudopotentials,
    )


def stand_in(translation, problem=None) -> SimpleNamespace:
    """What a prepared system tells the reporter of the domain, without a grid."""

    problem = translation.problem if problem is None else problem
    atoms = (
        center_cluster_geometry(problem.atoms)
        if problem.recenter_geometry
        else tuple(problem.atoms)
    )
    potentials = load_pseudopotentials(
        problem.pseudopotentials, xc_functional=problem.scf.xc_functional
    )
    electrons = ionic_charge(atoms, potentials) - problem.scf.net_charge
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        plan = plan_hartree_boundary(
            problem.hartree,
            problem.grid,
            *valence_point_charges(atoms, potentials, electrons),
        )
    return SimpleNamespace(
        input=problem,
        atoms=atoms,
        pseudopotentials=potentials,
        electron_count=electrons,
        hartree_boundary=plan,
    )


def nanodiamond(radius: bool):
    text, pseudopotentials = source_of(NANODIAMOND, BASE_PSEUDOPOTENTIALS)
    return parse_text(
        text if radius else parsed_inputs.without_radius(text), pseudopotentials
    )


class RuleArithmeticTests(unittest.TestCase):
    def test_table_integrals_use_the_radial_quadrature_of_the_tree(self) -> None:
        generator = np.random.default_rng(3)
        radii = np.cumsum(generator.uniform(0.01, 0.2, size=200))
        values = generator.normal(size=200)
        self.assertAlmostEqual(
            float(domain._quadrature_weights(radii) @ values),
            parsec_radial_integral(radii, values),
            delta=1.0e-13,
        )

    def test_a_sum_that_is_a_lattice_value_stays_on_it(self) -> None:
        # r_max + 3 ang typed on the lattice: the ceiling of the rounded sum
        # used to move 22 of these 400 up by 0.1 ang.
        for step in range(1, 401):
            radius = (0.1 * step + 3.0) * ANGSTROM_TO_BOHR
            self.assertEqual(
                domain.lattice_radius(radius), Decimal(step + 30) / 10, step
            )
        self.assertEqual(
            domain.lattice_radius(5.0000001 * ANGSTROM_TO_BOHR), Decimal("5.1")
        )
        self.assertEqual(
            domain.lattice_radius(4.925 * ANGSTROM_TO_BOHR, Decimal("0.025")),
            Decimal("4.925"),
        )

    def test_lengths_may_be_numpy_scalars(self) -> None:
        # What a caller computes with NumPy is a scalar of its own, and
        # GridSettings keeps the radius it is given.  repr() of one reads
        # "np.float64(31.7)" from NumPy 2 on, which is no decimal number.
        self.assertEqual(domain.lattice_radius(np.float64(31.7)), Decimal("16.8"))
        self.assertEqual(domain.lattice_radius(np.float32(8.5)), domain.lattice_radius(8.5))
        hydrogen = [Atom("H", (0.0, 0.0, 0.0))]
        potentials = load_pseudopotentials(
            {"H": SpeciesPotential(BASE_PSEUDOPOTENTIALS / "H_POTRE.DAT", 0)}
        )
        # The stencil of a coarse grid sets this radius from the spacing.
        plain = domain.default_domain(hydrogen, potentials, 1.0, spacing=1.2, budget=1.0)
        scalar = domain.default_domain(
            hydrogen,
            potentials,
            np.float64(1.0),
            spacing=np.float64(1.2),
            budget=np.float64(1.0),
        )
        self.assertEqual(
            (scalar.radius_text, scalar.set_by, scalar.radius, scalar.tolerance_text),
            (plain.radius_text, "stencil", plain.radius, plain.tolerance_text),
        )
        fit = domain.WallFit(
            prefactor=1.8e-4, decay=0.9, energy=1.0e-4, rms=0.02, at_scan_end=False, shells=8
        )
        self.assertEqual(
            domain.radius_for_budget(
                fit, np.float64(9.0), hydrogen, potentials, np.float64(1.0),
                spacing=np.float64(0.4), highest_occupied=np.float64(-0.3),
            ),
            domain.radius_for_budget(
                fit, 9.0, hydrogen, potentials, 1.0, spacing=0.4, highest_occupied=-0.3
            ),
        )

    def test_every_lattice_text_reads_back_to_the_radius_of_the_rule(self) -> None:
        self.assertEqual(domain.ANGSTROM, ANGSTROM_TO_BOHR)
        for step in range(10, 20001):
            value = Decimal(step) * Decimal("0.1")
            self.assertEqual(
                _physical_length(f"{value} ang", label="radius"),
                float(value) * domain.ANGSTROM,
            )
        self.assertEqual(
            _energy_rydberg("1.0e-03 Ry", label="tolerance"),
            _energy_rydberg("1e-3 Ry", label="tolerance"),
        )

    def test_tolerance_tightens_with_the_electron_count(self) -> None:
        vacuum = 4.5 * domain.ANGSTROM
        for case, electrons in BENCHMARK_ELECTRONS.items():
            tolerance = domain.auto_boundary_tolerance(electrons, 1.0e-3, vacuum)
            self.assertEqual(f"{tolerance:.1e} Ry", BENCHMARK[case][1], case)
            # The bound K sqrt(N_e) tau is then within the boundary share.
            self.assertLessEqual(
                domain.BOUNDARY_CONSTANT * math.sqrt(electrons) * tolerance, 5.0e-4
            )
        # Below 4 ang of vacuum the constant doubles: k06 at 3.8 ang.
        self.assertEqual(
            domain.auto_boundary_tolerance(3480, 1.0e-3, 3.8 * domain.ANGSTROM),
            6.7e-4,
        )
        self.assertEqual(
            domain.auto_boundary_tolerance(39368, 1.0e-3, 4.0 * domain.ANGSTROM),
            4.0e-4,
        )
        self.assertEqual(
            domain.auto_boundary_tolerance(39368, 1.0e-3, 3.99 * domain.ANGSTROM),
            2.0e-4,
        )
        # It never exceeds the fixed default and scales with the budget.
        self.assertEqual(domain.auto_boundary_tolerance(32, 1.0e-3, vacuum), 1.0e-3)
        self.assertEqual(domain.auto_boundary_tolerance(32, 1.0, vacuum), 1.0e-3)
        self.assertEqual(
            domain.auto_boundary_tolerance(39368, 1.0e-4, vacuum), 4.0e-5
        )

    def test_series_bound_is_never_below_the_estimate(self) -> None:
        # One charge: the bound is the estimate, the geometric series in the
        # direction of the charge.
        position, charge, radius = np.array([[0.0, 0.0, 8.1]]), np.array([3.0]), 9.0
        estimate = estimate_omitted_potential(position, charge, radius)
        for order in (9, 30, 60):
            self.assertAlmostEqual(
                domain.series_bound(position, charge, radius, order) / estimate[order],
                1.0,
                delta=1.0e-9,
            )
        self.assertEqual(domain.series_bound(position, charge, 8.0, 60), math.inf)
        translation = nanodiamond(radius=True)
        system = stand_in(translation)
        positions, charges = valence_point_charges(
            system.atoms, system.pseudopotentials, system.electron_count
        )
        for radius in (30.0, 31.747411546609168, 34.0):
            estimate = estimate_omitted_potential(positions, charges, radius)
            for order in (9, 19, 40, 60):
                self.assertGreaterEqual(
                    domain.series_bound(positions, charges, radius, order),
                    estimate[order],
                )
        # At the radius of the rule the bound spares this cluster the estimate.
        self.assertLess(
            domain.series_bound(positions, charges, 31.747411546609168, 60), 1.0e-3
        )

    def test_wall_estimate_falls_with_the_radius(self) -> None:
        system = stand_in(nanodiamond(radius=True))
        tables, reasons = domain.free_atom_tables(system.pseudopotentials)
        self.assertEqual(reasons, {})
        positions = np.array([atom.position for atom in system.atoms])
        symbols = [atom.symbol for atom in system.atoms]
        largest = np.linalg.norm(positions, axis=1).max()
        previous = math.inf
        for vacuum in np.arange(2.0, 9.01, 0.25):
            wall, parts = domain.wall_energy(
                positions, symbols, tables, largest + vacuum * domain.ANGSTROM
            )
            self.assertLess(wall, previous)
            self.assertAlmostEqual(sum(parts.values()), wall, delta=1.0e-18)
            previous = wall
        # The estimator of the measurements (a script that is not in this
        # tree: every atom, the trapezoid rule) tabulates twice the wall
        # energy: 8.769e-4, 2.294e-4 and 5.160e-2 Ry at 16.8, 17.3 and
        # 15.3 ang.
        for radius, twice in ((16.8, 8.769e-4), (17.3, 2.294e-4), (15.3, 5.160e-2)):
            wall, _parts = domain.wall_energy(
                positions, symbols, tables, radius * domain.ANGSTROM
            )
            self.assertAlmostEqual(2.0 * wall / twice, 1.0, delta=5.0e-4)
        # Carbon tetrafluoride of the examples: 9.172e-4, 1.743e-5 and
        # 1.578e-2 Ry at 4.9 ang, the 6.335 ang of its input, and 4.0 ang.
        text, folder = source_of(
            REPOSITORY / "examples/0_CH4_CF4/python_pbe/CF4/IS/parsec.in", None
        )
        system = stand_in(parse_text(text, folder))
        tables, _reasons = domain.free_atom_tables(system.pseudopotentials)
        positions = np.array([atom.position for atom in system.atoms])
        symbols = [atom.symbol for atom in system.atoms]
        for radius, twice in (
            (4.9, 9.172e-4),
            (6.334740118722521, 1.743e-5),
            (4.0, 1.578e-2),
        ):
            wall, _parts = domain.wall_energy(
                positions, symbols, tables, radius * domain.ANGSTROM
            )
            self.assertAlmostEqual(2.0 * wall / twice, 1.0, delta=5.0e-4)

    def test_free_atom_density_is_used_only_if_complete(self) -> None:
        # The hydrogen file of the tests ends at 1.60 bohr and holds a ninth
        # of its electron: neither its table nor its wavefunction passes.
        hydrogen = read_parsec_pseudopotential(DATA / "H_POTRE.DAT")
        tables, reasons = domain.free_atom_tables({"H": hydrogen})
        self.assertEqual(tables, {})
        self.assertRegex(
            reasons["H"],
            r"no complete free-atom density \(valence table 0\.\d+ e; "
            r"pseudo-wavefunctions 0\.\d+ e; occupations 1 e, table ends at "
            r"1\.60 bohr\)",
        )
        # A file generated for an ion holds the charge of that ion.
        folder = REPOSITORY / "examples" / "0_CH4_CF4" / "python_pbe" / "pseudopotentials"
        core_hole = read_parsec_pseudopotential(folder / "C-1s_POTRE.DAT")
        self.assertEqual(core_hole.ionic_charge, 5.0)
        tables, reasons = domain.free_atom_tables({"C-1s": core_hole})
        self.assertEqual(
            (tables["C-1s"].source, tables["C-1s"].charge), ("valence table", 4.0)
        )
        self.assertAlmostEqual(tables["C-1s"].integral, 4.0, delta=1.0e-3)
        # A table that does not carry the charge gives way to the
        # wavefunctions, which must pass the same test.
        carbon = read_parsec_pseudopotential(BASE_PSEUDOPOTENTIALS / "C_POTRE.DAT")
        emptied = replace(carbon, valence_density=0.4 * carbon.valence_density)
        tables, _reasons = domain.free_atom_tables({"C": emptied})
        self.assertEqual(tables["C"].source, "pseudo-wavefunctions")
        ionic = replace(emptied, channel_occupations={0: 2.0, 1: 1.0})
        tables, _reasons = domain.free_atom_tables({"C": ionic})
        self.assertEqual(tables["C"].charge, 3.0)


class RuleOnInputsTests(unittest.TestCase):
    def assert_rule(self, name, translation, expected) -> None:
        choice = translation.domain
        self.assertIsNotNone(choice, name)
        system = stand_in(translation)
        self.assertEqual(
            (
                choice.radius_text,
                choice.tolerance_text,
                system.hartree_boundary.order,
                system.hartree_boundary.engaged,
                choice.set_by,
                sorted(symbol for symbol, _reason in choice.without_table),
            ),
            (
                expected["radius"],
                expected["tolerance"],
                expected["order"],
                expected["engaged"],
                expected["set_by"],
                expected["without_table"],
            ),
            name,
        )
        # The reference script integrates with the trapezoid rule.
        self.assertAlmostEqual(choice.wall, expected["wall"], delta=1.0e-5 * choice.wall_share, msg=name)
        self.assertAlmostEqual(choice.extra_electrons, expected["extra"], delta=1.0e-3, msg=name)
        problem = translation.problem
        self.assertEqual(problem.grid.radius, choice.radius, name)
        self.assertEqual(problem.hartree.boundary_tolerance, choice.boundary_tolerance)
        self.assertEqual(translation.boundary_tolerance_from, "rule")
        self.assertGreater(translation.domain_rule_seconds, 0.0)

    def test_shipped_inputs_without_their_radius(self) -> None:
        for name, (path, pseudopotentials) in parsed_inputs.shipped_inputs().items():
            text, folder = source_of(path, pseudopotentials)
            with self.subTest(input=name):
                translation = parse_text(parsed_inputs.without_radius(text), folder)
                self.assert_rule(name, translation, REFERENCE["inputs"][name])

    def test_the_smallest_benchmark_cluster(self) -> None:
        translation = nanodiamond(radius=False)
        choice = translation.domain
        self.assertEqual(
            (choice.radius_text, choice.tolerance_text, choice.set_by),
            (BENCHMARK["k06"][0], BENCHMARK["k06"][1], "atomic densities"),
        )
        self.assertEqual(stand_in(translation).hartree_boundary.order, BENCHMARK["k06"][2])
        self.assertEqual(choice.electrons, BENCHMARK_ELECTRONS["k06"])
        self.assertAlmostEqual(choice.vacuum / domain.ANGSTROM, 4.505, delta=5.0e-4)
        self.assertLessEqual(choice.wall, choice.wall_share)
        self.assertEqual(choice.estimates, 0)
        self.assertEqual(translation.warnings, ())

    def assert_pasted(self, text: str, folder: Path) -> None:
        """An input without a radius, and the same with the two printed lines."""

        with input_folder(text) as path:
            roots = {"<input>": path.parent, "<repository>": REPOSITORY}
            chosen = parse_parsec_input(path, pseudopotential_directory=folder)
            printed = domain_header_lines(chosen)[-2:]
            self.assertRegex(printed[0], r"^       Boundary_Sphere_Radius: ")
            self.assertRegex(printed[1], r"^       Hartree_Boundary_Tolerance: ")
            path.write_bytes((text + "\n".join(printed) + "\n").encode("utf-8"))
            pasted = parse_parsec_input(path, pseudopotential_directory=folder)
            self.assertIsNone(pasted.domain)
            self.assertEqual(pasted.domain_rule_seconds, 0.0)
            self.assertEqual(pasted.boundary_tolerance_from, "input")
            self.assertEqual(
                pasted.problem.grid.radius.hex(), chosen.problem.grid.radius.hex()
            )
            self.assertEqual(
                pasted.problem.hartree.boundary_tolerance.hex(),
                chosen.problem.hartree.boundary_tolerance.hex(),
            )
            for settings in parsed_inputs.SETTINGS:
                self.assertEqual(
                    getattr(pasted.problem, settings),
                    getattr(chosen.problem, settings),
                )
            # Everything the reference cache key hashes, and with it what
            # the driver prepares; the warning of the rule aside.
            record = parsed_inputs.problem_record(chosen, roots)
            record["translation"][0] = [
                warning
                for warning in record["translation"][0]
                if "free atoms hold" not in warning
            ]
            self.assertEqual(parsed_inputs.problem_record(pasted, roots), record)
            self.assertEqual(
                stand_in(pasted).hartree_boundary.order,
                stand_in(chosen).hartree_boundary.order,
            )

    def test_pasting_the_printed_lines_gives_the_same_problem(self) -> None:
        cases = dict(parsed_inputs.shipped_inputs())
        constructed = parsed_inputs.constructed_inputs()
        for name in list(cases) + list(constructed):
            if name in cases:
                text, folder = source_of(*cases[name])
            else:
                text, folder = constructed[name]
            with self.subTest(input=name):
                self.assert_pasted(parsed_inputs.without_radius(text), folder)

    def test_electrons_no_free_atom_holds_stretch_the_vacuum(self) -> None:
        constructed = parsed_inputs.constructed_inputs()
        for name in (
            "fluoride anion",
            "methane with the core-hole carbon, neutral",
        ):
            text, folder = constructed[name]
            translation = parse_text(parsed_inputs.without_radius(text), folder)
            self.assert_rule(name, translation, REFERENCE["constructed"][name])
            choice = translation.domain
            self.assertEqual(choice.set_by, "electrons no free atom holds")
            self.assertAlmostEqual(choice.extra_electrons, 1.0, delta=1.0e-9)
            # max(1.6 v, v + 2.5 ang) of the vacuum the free atoms ask for.
            before = choice.wall_vacuum / domain.ANGSTROM
            self.assertGreaterEqual(
                choice.vacuum / domain.ANGSTROM, max(1.6 * before, before + 2.5) - 1.0e-9
            )
            self.assertLess(
                choice.vacuum / domain.ANGSTROM, max(1.6 * before, before + 2.5) + 0.1
            )
            (warning,) = translation.warnings
            self.assertRegex(
                warning,
                r"^1 electron\(s\) more than the free atoms hold: their tail is "
                r"unknown before the SCF; the vacuum was raised from \d\.\d to "
                r"\d\.\d ang\. This does not guarantee Domain_Energy_Tolerance\.",
            )
            lines = "\n".join(domain_header_lines(translation))
            self.assertIn(
                " --- Energy the sphere adds: no estimate before the SCF, the "
                "system holds 1 electron(s) more than its free atoms",
                lines,
            )
            self.assertIn(" --- Radius set by: electrons no free atom holds", lines)
        # The shipped cation of the same species: its table holds what the
        # system has, and nothing is stretched.
        text, folder = source_of(
            REPOSITORY / "examples/0_CH4_CF4/python_pbe/CH4/FS_1s/parsec.in", None
        )
        cation = parse_text(parsed_inputs.without_radius(text), folder).domain
        self.assertEqual((cation.radius_text, cation.set_by), ("5.2 ang", "atomic densities"))
        self.assertAlmostEqual(cation.extra_electrons, 0.0, delta=1.0e-9)
        self.assertIn(("C-1s", "valence table", 4.0), cation.tables)

    def test_species_without_a_complete_density_gets_the_guess(self) -> None:
        text, folder = source_of(DATA / "H2_parsec.in", None)
        translation = parse_text(parsed_inputs.without_radius(text), folder)
        choice = translation.domain
        # 0.375 ang + 7.5 ang, up to the lattice; the minimum vacuum alone
        # gave 3.4 ang, where the sphere adds 3.3e-3 Ry.
        self.assertEqual(
            (choice.radius_text, choice.set_by, choice.wall),
            ("7.9 ang", "no atomic density for H", 0.0),
        )
        self.assertEqual(choice.tables, ())
        lines = "\n".join(domain_header_lines(translation))
        self.assertIn(
            " --- Energy the sphere adds: no estimate before the SCF, no "
            "species has a complete free-atom density",
            lines,
        )
        self.assertRegex(
            lines,
            r" --- NOTE: H has no complete free-atom density \(.*\): its atoms "
            r"get 7\.5 ang of vacuum, which is a guess",
        )
        self.assertNotIn("0.0E+00", lines)

    def test_no_grid_point_lies_on_the_sphere(self) -> None:
        constructed = parsed_inputs.constructed_inputs()
        for spacing, radius, step in (
            ("0.1", "4.95 ang", "+0.05"),
            ("0.05", "4.925 ang", "+0.025"),
            ("0.2", "4.9 ang", None),
        ):
            name = f"CF4 on a grid of {spacing} ang through the origin"
            text, folder = constructed[name]
            translation = parse_text(parsed_inputs.without_radius(text), folder)
            self.assert_rule(name, translation, REFERENCE["constructed"][name])
            choice = translation.domain
            self.assertEqual(choice.radius_text, radius)
            self.assertEqual(translation.problem.grid.shift, (0.0, 0.0, 0.0))
            if step is None:
                self.assertEqual(choice.set_by, "atomic densities")
            else:
                self.assertEqual(
                    choice.set_by,
                    f"atomic densities ({step} ang: a grid point lay on the sphere)",
                )
            squared = (choice.radius / translation.problem.grid.spacing) ** 2
            self.assertGreater(abs(squared - round(squared)) / squared, 1.0e-9)
        # The shifted grid of the same molecule needs no step.
        text, folder = source_of(
            REPOSITORY / "examples/0_CH4_CF4/python_pbe/CF4/IS/parsec.in", None
        )
        shifted = parse_text(
            parsed_inputs.without_radius(text).replace(
                "Grid_Spacing: 0.07 ang", "Grid_Spacing: 0.1 ang"
            ),
            folder,
        )
        self.assertEqual(shifted.domain.radius_text, "4.9 ang")

    def rule(self, atoms, **options):
        potentials = load_pseudopotentials(
            {
                symbol: SpeciesPotential(
                    BASE_PSEUDOPOTENTIALS / f"{symbol}_POTRE.DAT", 0
                )
                for symbol in {atom.symbol for atom in atoms}
            }
        )
        electrons = options.pop("electrons", ionic_charge(atoms, potentials))
        options.setdefault("spacing", 0.4)
        return domain.default_domain(atoms, potentials, electrons, **options)

    def test_radius_keeps_the_minimum_vacuum_and_the_stencil(self) -> None:
        hydrogen = [Atom("H", (0.0, 0.0, 0.0))]
        # A budget that wide leaves the 3 ang of the minimum vacuum.
        loose = self.rule(hydrogen, budget=1.0)
        self.assertEqual((loose.radius_text, loose.set_by), ("3.0 ang", "minimum vacuum"))
        # Seven points of 1.2 bohr beyond the atom are more than that.
        coarse = self.rule(hydrogen, budget=1.0, spacing=1.2)
        self.assertEqual((coarse.radius_text, coarse.set_by), ("4.5 ang", "stencil"))
        self.assertGreaterEqual(coarse.radius, 7 * 1.2)
        self.assertLess(coarse.radius - 0.1 * domain.ANGSTROM, 7 * 1.2)
        # The default budget asks for more than either.
        default = self.rule(hydrogen)
        self.assertEqual(default.set_by, "atomic densities")
        self.assertGreater(default.radius, loose.radius)
        self.assertLessEqual(default.wall, default.wall_share)
        tighter = self.rule(hydrogen, budget=1.0e-4)
        self.assertGreater(tighter.radius, default.radius)
        self.assertLessEqual(tighter.wall, 5.0e-5)
        with self.assertRaisesRegex(ValueError, "Domain_Energy_Tolerance"):
            self.rule(hydrogen, budget=0.0)
        # The helper of callers that build a SinglePointInput themselves.
        from parsec_python import Hartree

        self.assertIs(Hartree.default_domain, domain.default_domain)
        self.assertIsInstance(default, Hartree.DomainChoice)
        grid = GridSettings(spacing=0.4, radius=default.radius)
        self.assertEqual(
            grid.radius, _physical_length(default.radius_text, label="radius")
        )

    def test_radius_grows_where_the_largest_order_misses_the_tolerance(self) -> None:
        # One hydrogen atom 100 ang from the centre: the omitted potential
        # is 2 (d/R)**61 / (R - d) Ry, 1.7e-2 Ry at the radius of the wall.
        distance = 100.0 * domain.ANGSTROM
        atom = [Atom("H", (0.0, 0.0, distance))]
        choice = self.rule(atom)
        self.assertEqual(choice.set_by, "multipole order")
        self.assertEqual(choice.tolerance_text, "1.0e-03 Ry")

        def omitted(radius):
            return 2.0 * (distance / radius) ** 61 / (radius - distance)

        self.assertLessEqual(omitted(choice.radius), 1.0e-3)
        self.assertGreater(omitted(choice.radius - 0.1 * domain.ANGSTROM), 1.0e-3)
        self.assertGreater(choice.vacuum / domain.ANGSTROM, 7.0)
        self.assertGreater(choice.estimates, 0)
        plan = plan_hartree_boundary(
            HartreeSettings(boundary_tolerance=choice.boundary_tolerance),
            GridSettings(spacing=0.4, radius=choice.radius),
            np.array([[0.0, 0.0, distance]]),
            np.array([1.0]),
        )
        self.assertEqual((plan.order, plan.cap_reached), (60, False))
        # A tolerance of the input decides instead of the rule's own, and the
        # direct Coulomb sum needs no order at all.
        tight = self.rule(atom, boundary_tolerance=1.0e-4)
        self.assertIsNone(tight.tolerance_text)
        self.assertGreater(tight.radius, choice.radius)
        self.assertLessEqual(omitted(tight.radius), 1.0e-4)
        exact = self.rule(atom, multipole_boundary=False)
        self.assertEqual(exact.set_by, "atomic densities")
        self.assertEqual((exact.tolerance_text, exact.boundary_tolerance, exact.estimates), (None, None, 0))
        self.assertLess(exact.radius, choice.radius)

    @needs_measurements
    def test_benchmark_clusters(self) -> None:
        for case, (radius, tolerance, order) in BENCHMARK.items():
            folder = Path(MEASUREMENTS) / "recipe_inputs" / f"{case}_ff4_fd30_d15" / "python_sad"
            text, folder = source_of(folder / "parsec.in", None)
            with self.subTest(case=case):
                translation = parse_text(parsed_inputs.without_radius(text), folder)
                choice = translation.domain
                self.assertEqual(
                    (choice.radius_text, choice.tolerance_text, choice.set_by),
                    (radius, tolerance, "atomic densities"),
                )
                plan = stand_in(translation).hartree_boundary
                self.assertEqual((plan.order, plan.engaged, plan.atomic_tail), (order, True, True))
                self.assertEqual(choice.electrons, BENCHMARK_ELECTRONS[case])
                self.assertLessEqual(choice.wall, choice.wall_share)
                self.assertGreater(choice.wall, 0.8 * choice.wall_share)
                # The bound of the boundary values is within their share.
                energy = domain.boundary_energy(plan, choice.electrons, choice.vacuum)
                self.assertLessEqual(energy.bound, 5.0e-4)
                self.assertLessEqual(choice.estimates, 1)
                self.assert_pasted(parsed_inputs.without_radius(text), folder)

    @needs_measurements
    def test_systems_without_hydrogen_at_the_surface(self) -> None:
        for name, (radius, vacuum, wall, shares) in GENERAL.items():
            folder = Path(MEASUREMENTS) / "general_inputs" / name
            text, folder = source_of(folder / "parsec.in", None)
            with self.subTest(system=name):
                translation = parse_text(parsed_inputs.without_radius(text), folder)
                choice = translation.domain
                self.assertEqual(
                    (choice.radius_text, choice.tolerance_text),
                    (radius, "1.0e-03 Ry"),
                )
                self.assertAlmostEqual(choice.vacuum / domain.ANGSTROM, vacuum, delta=5.1e-3)
                plan = stand_in(translation).hartree_boundary
                self.assertEqual((plan.order, plan.engaged), (9, name == "c35_bare"))
                self.assertEqual(choice.estimates, 0)
                lines = "\n".join(domain_header_lines(translation))
                if wall is None:
                    # Cl-: 4.4 ang for the neutral atom, stretched by 1.6.
                    self.assertEqual(choice.set_by, "electrons no free atom holds")
                    self.assertAlmostEqual(choice.wall_vacuum / domain.ANGSTROM, 4.4, delta=1.0e-9)
                    self.assertEqual(len(translation.warnings), 1)
                else:
                    self.assertEqual(choice.set_by, "atomic densities")
                    self.assertEqual(f"{choice.wall:.1e}", wall)
                    self.assertIn(f"share of the estimate: {shares}", lines)
                    self.assertEqual(translation.warnings, ())
                self.assert_pasted(parsed_inputs.without_radius(text), folder)


class ParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = parsed_inputs.BASE_INPUT
        self.auto = parsed_inputs.without_radius(self.text)

    def parse(self, text):
        return parse_text(text, BASE_PSEUDOPOTENTIALS)

    def test_radius_absent_and_auto_are_the_same(self) -> None:
        absent = self.parse(self.auto)
        for value in ("auto", "Auto", " AUTO "):
            written = self.parse(self.auto + f"Boundary_Sphere_Radius: {value}\n")
            self.assertEqual(written.domain.radius_text, absent.domain.radius_text)
            self.assertEqual(written.problem.grid, absent.problem.grid)
            self.assertEqual(written.problem.hartree, absent.problem.hartree)
        self.assertEqual(absent.domain.radius_text, "6.6 ang")
        self.assertEqual(absent.domain_energy_tolerance, 1.0e-3)
        # The settings of the grid are those of an input with this radius.
        given = self.parse(self.auto + "Boundary_Sphere_Radius: 6.6 ang\n")
        self.assertEqual(given.problem.grid, absent.problem.grid)
        self.assertIsNone(given.domain)
        self.assertEqual(given.boundary_tolerance_from, "default")

    def test_domain_energy_tolerance_is_the_one_knob(self) -> None:
        radii = {}
        for budget in ("2e-3 Ry", "1e-3 Ry", "1e-4 Ry", "6.8e-3 eV"):
            translation = self.parse(self.auto + f"Domain_Energy_Tolerance: {budget}\n")
            radii[budget] = translation.domain.radius
            self.assertLessEqual(
                translation.domain.wall, 0.5 * translation.domain_energy_tolerance
            )
        self.assertGreater(radii["1e-4 Ry"], radii["1e-3 Ry"])
        self.assertGreater(radii["1e-3 Ry"], radii["2e-3 Ry"])
        self.assertLess(radii["6.8e-3 eV"], radii["1e-4 Ry"])
        # It does not touch the SCF criterion, and an input with a radius
        # only keeps it for its notes.
        self.assertEqual(
            self.parse(self.auto).problem.scf.convergence_criterion, 1.0e-4
        )
        given = self.parse(self.text + "Domain_Energy_Tolerance: 1e-4 Ry\n")
        plain = self.parse(self.text)
        self.assertEqual(given.domain_energy_tolerance, 1.0e-4)
        self.assertIsNone(given.domain)
        for settings in parsed_inputs.SETTINGS:
            self.assertEqual(getattr(given.problem, settings), getattr(plain.problem, settings))
        for value in ("0 Ry", "-1e-3 Ry"):
            with self.assertRaisesRegex(ParsecInputError, "Domain_Energy_Tolerance must be positive"):
                self.parse(self.text + f"Domain_Energy_Tolerance: {value}\n")
        with self.assertRaisesRegex(ParsecInputError, "unsupported energy unit"):
            self.parse(self.text + "Domain_Energy_Tolerance: 1e-3 furlong\n")

    def test_tolerance_of_the_input_wins(self) -> None:
        chosen = self.parse(self.auto)
        self.assertEqual(chosen.boundary_tolerance_from, "rule")
        explicit = self.parse(self.auto + "Hartree_Boundary_Tolerance: 2e-5 Ry\n")
        self.assertEqual(explicit.problem.hartree.boundary_tolerance, 2.0e-5)
        self.assertEqual(explicit.boundary_tolerance_from, "input")
        self.assertIsNone(explicit.domain.tolerance_text)
        self.assertEqual(explicit.domain.radius_text, chosen.domain.radius_text)
        # One line then gives the domain again.
        lines = domain_header_lines(explicit)
        self.assertRegex(lines[-2], r"^ --- This line, beside the others of the input, reproduces")
        self.assertEqual(lines[-1], "       Boundary_Sphere_Radius: 6.6 ang")
        again = self.parse(self.auto + "Hartree_Boundary_Tolerance: 2e-5 Ry\n" + lines[-1] + "\n")
        self.assertEqual(again.problem.grid, explicit.problem.grid)
        self.assertEqual(again.problem.hartree, explicit.problem.hartree)

    def test_tolerance_auto_beside_a_radius(self) -> None:
        # 21 * 4 ... benzene has 30 electrons: the rule's tolerance is the
        # default.  A budget ten times tighter tightens it.
        translation = self.parse(
            self.text
            + "Hartree_Boundary_Tolerance: auto\nDomain_Energy_Tolerance: 1e-5 Ry\n"
        )
        self.assertIsNone(translation.domain)
        self.assertEqual(translation.boundary_tolerance_from, "rule")
        self.assertGreater(translation.domain_rule_seconds, 0.0)
        # 6.0 - 2.47 ang of vacuum is below 4 ang: 5e-6 / (1.25e-2 sqrt(30)).
        self.assertEqual(translation.problem.hartree.boundary_tolerance, 7.3e-5)
        self.assertEqual(translation.problem.grid, self.parse(self.text).problem.grid)
        self.assertEqual(
            self.parse(self.text + "Hartree_Boundary_Tolerance: auto\n").problem.hartree,
            self.parse(self.text).problem.hartree,
        )
        # With the direct Coulomb sum there is nothing to tolerate.
        direct = self.parse(self.text + "Hartree_Boundary_Tolerance: auto\nFull_Hartree: true\n")
        self.assertEqual(direct.problem.hartree.boundary_tolerance, 1.0e-3)
        self.assertEqual(direct.boundary_tolerance_from, "input")

    def test_radius_of_the_rule_needs_the_controlled_boundary(self) -> None:
        message = (
            "Boundary_Sphere_Radius is left to the default rule, and the default "
            "radius is chosen for the controlled Hartree boundary: with "
            "Hartree_Boundary_Tolerance: off or Hartree_Atomic_Tail: off give "
            "Boundary_Sphere_Radius"
        )
        for line in ("Hartree_Boundary_Tolerance: off", "Hartree_Atomic_Tail: off"):
            with self.assertRaises(ParsecInputError) as raised:
                self.parse(self.auto + line + "\n")
            self.assertEqual(str(raised.exception), message)
            # With a radius the line means what it did.
            self.assertIsNone(self.parse(self.text + line + "\n").domain)
            # The direct Coulomb sum has no boundary to control: the rule
            # chooses the radius alone.
            exact = self.parse(self.auto + line + "\nFull_Hartree: true\n")
            self.assertEqual(exact.domain.radius_text, "6.6 ang")
            self.assertIsNone(exact.domain.tolerance_text)
            self.assertNotEqual(exact.boundary_tolerance_from, "rule")
        full = self.parse(self.auto + "Full_Hartree: true\n")
        self.assertEqual(full.problem.hartree.boundary_tolerance, 1.0e-3)
        self.assertEqual(full.boundary_tolerance_from, "default")

    def test_box_is_untouched(self) -> None:
        box = self.auto.replace(
            "Cluster_Domain_Shape: sphere", "Cluster_Domain_Shape: box"
        ) + "begin Domain_Shape_Parameters\n 24 24 24\nend Domain_Shape_Parameters\n"
        translation = self.parse(box)
        self.assertIsNone(translation.domain)
        self.assertEqual(translation.problem.grid.box_lengths, (24.0, 24.0, 24.0))
        self.assertEqual(translation.domain_rule_seconds, 0.0)
        self.assertEqual(domain_header_lines(translation), [])

    def test_pasted_lines_that_are_no_input_lines_still_stop(self) -> None:
        # What a careless paste of the report produces.
        radius = "Boundary_Sphere_Radius: 6.6 ang"
        for text, pattern in (
            (
                self.auto + radius + " / Hartree_Boundary_Tolerance: 4.0e-04 Ry\n",
                "Boundary_Sphere_Radius: unsupported length unit",
            ),
            (
                self.auto + radius + " with Hartree_Boundary_Tolerance: 4.0e-04 Ry (order 50)\n",
                "Boundary_Sphere_Radius: unsupported length unit",
            ),
            (
                self.auto + radius + "\nHartree_Boundary_Tolerance: 4.0e-04 Ry would meet it\n",
                "Hartree_Boundary_Tolerance: unsupported energy unit",
            ),
            (self.auto + " --- " + radius + "\n", "unsupported or unknown PARSEC option"),
            (
                self.auto + radius + "\n" + radius + "\n",
                "duplicate boundary_sphere_radius values",
            ),
            (
                self.auto + "Domain_Energy_Tolerance: Boundary_Sphere_Radius: 5.3 ang would meet it\n",
                "Domain_Energy_Tolerance: unsupported energy unit",
            ),
            (
                self.auto + "Boundary_Sphere_Radius: auto\nBoundary_Sphere_Radius: auto\n",
                "duplicate boundary_sphere_radius values",
            ),
            (self.auto + "Boundary_Sphere_Radius: automatic\n", "expected a numeric value"),
        ):
            with self.assertRaisesRegex(ParsecInputError, pattern):
                self.parse(text)
        # An indented paste of the two lines is an input like any other.
        indented = self.parse(
            self.auto + "       " + radius + "\n       Hartree_Boundary_Tolerance: 4.0e-04 Ry\n"
        )
        self.assertEqual(indented.problem.hartree.boundary_tolerance, 4.0e-4)

    def test_system_without_electrons_stops_in_the_parser(self) -> None:
        with self.assertRaisesRegex(ParsecInputError, "no valence electrons"):
            self.parse(self.auto.replace("Net_Charges: 0 e", "Net_Charges: 30 e"))


class InputsWithARadiusTests(unittest.TestCase):
    """An input that gives its radius is parsed as it was before the rule."""

    def test_shipped_inputs(self) -> None:
        roots = {"<repository>": REPOSITORY}
        inputs = parsed_inputs.shipped_inputs()
        self.assertEqual(sorted(inputs), sorted(GOLDEN["inputs"]))
        for name, (path, pseudopotentials) in inputs.items():
            with self.subTest(input=name):
                translation = parse_parsec_input(
                    path, pseudopotential_directory=pseudopotentials
                )
                self.assertEqual(
                    parsed_inputs.problem_record(translation, roots),
                    GOLDEN["inputs"][name],
                )
                self.assertIsNone(translation.domain)
                self.assertEqual(translation.domain_rule_seconds, 0.0)
                self.assertEqual(translation.domain_energy_tolerance, 1.0e-3)
                self.assertEqual(translation.boundary_tolerance_from, "default")

    def test_periodic_inputs_are_kept_apart_from_those_with_a_sphere(self) -> None:
        roots = {"<repository>": REPOSITORY}
        periodic = parsed_inputs.periodic_inputs()
        self.assertIn("examples/3d_Si/parsec.in", periodic)
        self.assertFalse(set(periodic) & set(parsed_inputs.shipped_inputs()))
        for name, (path, pseudopotentials) in periodic.items():
            with self.subTest(input=name):
                # No radius line to take out, and no sphere to give one.
                text = path.read_text(encoding="utf-8")
                self.assertNotRegex(text, r"(?im)^[ \t]*Boundary_Sphere_Radius\b")
                translation = parse_parsec_input(
                    path, pseudopotential_directory=pseudopotentials
                )
                self.assertIsNone(translation.domain)
                self.assertEqual(translation.domain_rule_seconds, 0.0)
                self.assertFalse(hasattr(translation.problem.grid, "radius"))
                # The cell is part of what the reference cache key hashes:
                # only the "None" of a cluster is left out of a record.
                problem = translation.problem
                stretched = replace(
                    problem,
                    periodic_cell=replace(
                        problem.periodic_cell,
                        lattice_vectors=1.5 * problem.periodic_cell.lattice_vectors,
                    ),
                )
                self.assertNotEqual(
                    parsed_inputs.reference_cache_key(stretched, roots)[1],
                    parsed_inputs.reference_cache_key(problem, roots)[1],
                )

    def test_inputs_that_are_not_of_the_tree_are_left_out(self) -> None:
        # A checkout may hold example folders of its own.  Their inputs have
        # no record, and need not parse: this one has no pseudopotential
        # file beside it.
        shipped = parsed_inputs.shipped_inputs()
        periodic = parsed_inputs.periodic_inputs()
        self.assertEqual(sorted(shipped), sorted(GOLDEN["inputs"]))
        every = parsed_inputs._every_input()
        with tempfile.TemporaryDirectory() as directory:
            stray = Path(directory) / "parsec.in"
            stray.write_text(parsed_inputs.BASE_INPUT, encoding="utf-8")
            with self.assertRaisesRegex(ParsecInputError, "missing pseudopotential"):
                parse_parsec_input(stray)
            more = {**every, "examples/a_local_folder/parsec.in": (stray, None)}
            with patch.object(parsed_inputs, "_every_input", return_value=more):
                self.assertEqual(parsed_inputs.shipped_inputs(), shipped)
                self.assertEqual(parsed_inputs.periodic_inputs(), periodic)
        # An input of the records that the tree has lost is an error.
        fewer = dict(every)
        del fewer["examples/0d_benzene/parsec.in"]
        with patch.object(parsed_inputs, "_every_input", return_value=fewer):
            with self.assertRaisesRegex(AssertionError, "0d_benzene"):
                parsed_inputs.shipped_inputs()

    def test_mistakes_stop_with_the_same_words_in_the_same_order(self) -> None:
        self.assertEqual(
            sorted([*parsed_inputs.MALFORMED, "none"]), sorted(GOLDEN["malformed"])
        )
        here = Path.cwd()
        for name in GOLDEN["malformed"]:
            text = (
                parsed_inputs.BASE_INPUT
                if name == "none"
                else parsed_inputs.malformed_text(name)
            )
            with self.subTest(mistake=name), input_folder(text) as path:
                # The parser also looks for pseudopotentials where it runs.
                os.chdir(path.parent)
                try:
                    record = parsed_inputs.input_record(
                        path,
                        BASE_PSEUDOPOTENTIALS,
                        {"<repository>": REPOSITORY, "<input>": path.parent},
                    )
                finally:
                    os.chdir(here)
                self.assertEqual(record, GOLDEN["malformed"][name])
                self.assertEqual("error" in record, name != "none")

    @needs_measurements
    def test_benchmark_inputs(self) -> None:
        for case in BENCHMARK:
            folder = Path(MEASUREMENTS).resolve() / "recipe_inputs" / f"{case}_ff4_fd30_d15" / "python_sad"
            with self.subTest(case=case):
                self.assertEqual(
                    parsed_inputs.input_record(
                        folder / "parsec.in",
                        None,
                        {"<repository>": REPOSITORY, "<input>": folder},
                    ),
                    GOLDEN["benchmark"][case],
                )

    def test_estimate_of_the_boundary_is_made_once_for_a_given_radius(self) -> None:
        calls = []

        def counted(*arguments, **options):
            calls.append(arguments[2])
            return estimate_omitted_potential(*arguments, **options)

        text, folder = source_of(DATA / "H_cli_smoke.in", None)
        with (
            patch.object(boundary_module, "estimate_omitted_potential", counted),
            patch.object(domain, "estimate_omitted_potential", counted),
        ):
            given = parse_text(text, folder)
            self.assertEqual(calls, [])
            prepare_single_point(given.problem)
            self.assertEqual(calls, [4.0])
            del calls[:]
            # The rule of a small system makes none: the series bound tells
            # it that the largest order will do.
            chosen = parse_text(
                parsed_inputs.without_radius(text).replace(
                    "Grid_Spacing 0.8 bohr", "Grid_Spacing 1.2 bohr"
                ),
                folder,
            )
            self.assertEqual(calls, [])
            prepare_single_point(chosen.problem)
            self.assertEqual(calls, [chosen.domain.radius])


class ReportTests(unittest.TestCase):
    def test_header_of_a_radius_of_the_rule(self) -> None:
        translation = nanodiamond(radius=False)
        self.assertEqual(
            domain_header_lines(translation),
            [
                " --- Radius 16.8 ang chosen by the default rule 1 "
                "(Domain_Energy_Tolerance = 1.0E-03 Ry)",
                " --- Outermost atom: H at 12.295 ang from the centre; vacuum "
                "beyond it 4.505 ang",
                " --- Energy the sphere adds, estimated from the free atoms: "
                "4.4E-04 Ry (aim 5.0E-04 Ry; an estimate, not a bound)",
                " --- Free-atom densities: C valence table 4 e, H valence table "
                "1 e; share of the estimate: C 23 %, H 77 %",
                " --- Radius set by: atomic densities",
                " --- The rule covers the total energy and the occupied levels; "
                "empty levels need more vacuum.",
                " --- These two lines reproduce this domain bit for bit (with "
                "the same PARSEC_HARTREE_* switches):",
                "       Boundary_Sphere_Radius: 16.8 ang",
                "       Hartree_Boundary_Tolerance: 1.0e-03 Ry",
            ],
        )
        written = []
        ParsecTextReporter(written.append, translation).header()
        header = written[0].split("\n")
        start = header.index(" --- Radius is  31.747412 bohrs")
        self.assertEqual(header[start + 1 : start + 10], domain_header_lines(translation))
        self.assertIn(
            " Hartree boundary tolerance of the default rule is : 1.000000E-03"
            "  Ry   atomic tail : auto",
            header,
        )

    def test_header_of_a_radius_of_the_input_is_what_it_was(self) -> None:
        translation = nanodiamond(radius=True)
        self.assertEqual(domain_header_lines(translation), [])
        written = []
        ParsecTextReporter(written.append, translation).header()
        header = written[0].split("\n")
        start = header.index(" --- Radius is  32.692275 bohrs")
        self.assertEqual(header[start + 1], " Grid spacing is  0.377945 bohrs")
        self.assertIn(
            " Hartree boundary tolerance in the input is : 1.000000E-03  Ry"
            "   atomic tail : auto",
            header,
        )
        self.assertFalse(any("default rule" in line for line in header))

    def test_setup_lines_before_the_scf(self) -> None:
        chosen = nanodiamond(radius=False)
        lines, record = domain_setup_lines(stand_in(chosen), chosen)
        self.assertEqual(
            lines,
            [
                " --- Energy left by the boundary values at order 19 (tolerance "
                "1.0E-03 Ry, atomic tail): about 8.0E-05 Ry, calibrated bound "
                "2.5E-04 Ry (aim 5.0E-04 Ry)",
                "       calibration: hydrogen-terminated clusters, 176 to "
                "23,768 electrons",
            ],
        )
        self.assertEqual(
            {
                key: record[key]
                for key in (
                    "rule_version",
                    "radius_from",
                    "boundary_sphere_radius",
                    "hartree_boundary_tolerance",
                    "hartree_boundary_tolerance_from",
                    "set_by",
                    "outermost_atom",
                    "extra_electrons",
                    "species_without_table",
                    "free_atom_tables",
                )
            },
            dict(
                rule_version=1,
                radius_from="default rule",
                boundary_sphere_radius="16.8 ang",
                hartree_boundary_tolerance="1.0e-03 Ry",
                hartree_boundary_tolerance_from="rule",
                set_by="atomic densities",
                outermost_atom="H",
                extra_electrons=0.0,
                species_without_table={},
                free_atom_tables={
                    "C": dict(source="valence table", electrons=4.0),
                    "H": dict(source="valence table", electrons=1.0),
                },
            ),
        )
        self.assertEqual(record["radius_bohr"], chosen.problem.grid.radius)
        self.assertEqual(record["rule_seconds"], chosen.domain_rule_seconds)
        self.assertAlmostEqual(record["vacuum_angstrom"], 4.505, delta=5.0e-4)
        self.assertAlmostEqual(record["wall_estimate_ry"], 4.3846e-4, delta=1.0e-8)
        self.assertAlmostEqual(
            sum(record["wall_estimate_by_species"].values()), record["wall_estimate_ry"]
        )
        self.assertAlmostEqual(record["boundary"]["about_ry"], 8.03e-5, delta=1.0e-7)
        self.assertAlmostEqual(record["boundary"]["bound_ry"], 2.51e-4, delta=1.0e-6)

        given = nanodiamond(radius=True)
        lines, record = domain_setup_lines(stand_in(given), given)
        self.assertEqual(
            lines,
            [
                " --- Sphere of the input: outermost atom H at 12.295 ang from "
                "the centre; vacuum beyond it 5.005 ang",
                " --- Energy the sphere adds, estimated from the free atoms: "
                "1.1E-04 Ry (wall share of Domain_Energy_Tolerance 5.0E-04 Ry; "
                "an estimate, not a bound)",
                " --- Energy left by the boundary values at order 16 (tolerance "
                "1.0E-03 Ry, atomic tail): about 1.2E-04 Ry, calibrated bound "
                "3.6E-04 Ry (aim 5.0E-04 Ry)",
                "       calibration: hydrogen-terminated clusters, 176 to "
                "23,768 electrons",
            ],
        )
        self.assertEqual(
            (record["radius_from"], record["set_by"], record["boundary_sphere_radius"]),
            ("input", "input", None),
        )
        self.assertEqual(record["rule_seconds"], 0.0)
        self.assertAlmostEqual(record["wall_estimate_ry"], 1.1468e-4, delta=1.0e-8)

    def test_notes_of_a_radius_of_the_input(self) -> None:
        # A budget of 1e-4 Ry: the 5 ang sphere of the input misses the wall
        # share, and the fixed tolerance the boundary share.
        given = replace(nanodiamond(radius=True), domain_energy_tolerance=1.0e-4)
        lines, record = domain_setup_lines(stand_in(given), given)
        self.assertEqual(
            lines[2],
            " NOTE: the free-atom estimate exceeds the wall share. It is several "
            "times too large for some surfaces (hydrogen-terminated carbon): "
            "the estimate from the density after the SCF decides",
        )
        self.assertEqual(
            lines[-1],
            " NOTE: the calibrated bound exceeds the boundary share of "
            "Domain_Energy_Tolerance; Hartree_Boundary_Tolerance: 1.3e-04 Ry "
            "would meet it",
        )
        # That tolerance does meet it.
        tightened = replace(
            given.problem,
            hartree=replace(given.problem.hartree, boundary_tolerance=1.3e-4),
        )
        lines, record = domain_setup_lines(stand_in(given, tightened), given)
        self.assertLessEqual(record["boundary"]["bound_ry"], 5.0e-5)
        self.assertFalse(any("calibrated bound exceeds" in line for line in lines))
        # With Output_Level 2 the rule is run for the input as well.
        detailed = replace(nanodiamond(radius=True), output_level=2)
        lines, _record = domain_setup_lines(stand_in(detailed), detailed)
        self.assertEqual(
            lines[-3:],
            [
                " --- Without Boundary_Sphere_Radius the default rule 1 would "
                "give (set by atomic densities; wall estimate 4.4E-04 Ry):",
                "       Boundary_Sphere_Radius: 16.8 ang",
                "       Hartree_Boundary_Tolerance: 1.0e-03 Ry",
            ],
        )

    def test_note_of_a_radius_taken_from_the_block_after_an_scf(self) -> None:
        # The benchmark cluster at the 16.1 ang its runs print after the SCF:
        # the free atoms give six times the wall share there, the measured
        # energy is half of it.  The note does not send the reader back to
        # the larger radius of the rule.
        text, folder = source_of(NANODIAMOND, BASE_PSEUDOPOTENTIALS)
        second = parse_text(
            parsed_inputs.without_radius(text) + "Boundary_Sphere_Radius: 16.1 ang\n",
            folder,
        )
        lines, record = domain_setup_lines(stand_in(second), second)
        self.assertAlmostEqual(record["wall_estimate_ry"], 2.91e-3, delta=1.0e-5)
        self.assertEqual(
            lines[1:3],
            [
                " --- Energy the sphere adds, estimated from the free atoms: "
                "2.9E-03 Ry (wall share of Domain_Energy_Tolerance 5.0E-04 Ry; "
                "an estimate, not a bound)",
                " NOTE: the free-atom estimate exceeds the wall share. It is "
                "several times too large for some surfaces (hydrogen-terminated "
                "carbon): the estimate from the density after the SCF decides",
            ],
        )
        self.assertFalse(any("default rule" in line for line in lines))

    def test_estimate_names_the_species_it_leaves_out(self) -> None:
        # Methane with the carbon of the examples and the hydrogen file of
        # the tests, which holds no complete density: the hydrogen atoms are
        # the outermost, and the number is that of the carbon atom alone.
        methane = (
            "Boundary_Conditions: cluster\nCluster_Domain_Shape: sphere\n"
            "Grid_Spacing: 0.3 ang\nCoordinate_Unit: Cartesian_Ang\n"
            "States_Num: 8\nAtom_Types_Num: 2\n"
            "Atom_Type: C\nLocal_Component: p\n"
            "begin Atom_Coord\n 0 0 0\nend Atom_Coord\n"
            "Atom_Type: H\nLocal_Component: s\n"
            "begin Atom_Coord\n 0.63 0.63 0.63\n -0.63 -0.63 0.63\n"
            " -0.63 0.63 -0.63\n 0.63 -0.63 -0.63\nend Atom_Coord\n"
        )
        left_out = "; without H, which has no complete free-atom density)"
        with input_folder(methane) as path:
            (path.parent / "C_POTRE.DAT").write_bytes(
                (BASE_PSEUDOPOTENTIALS / "C_POTRE.DAT").read_bytes()
            )
            (path.parent / "H_POTRE.DAT").write_bytes((DATA / "H_POTRE.DAT").read_bytes())
            chosen = parse_parsec_input(path)
            header = domain_header_lines(chosen)
            path.write_bytes((methane + "Boundary_Sphere_Radius: 8.6 ang\n").encode("utf-8"))
            given = parse_parsec_input(path)
            setup, _record = domain_setup_lines(stand_in(given), given)
        self.assertEqual(
            (chosen.domain.radius_text, chosen.domain.set_by),
            ("8.6 ang", "no atomic density for H"),
        )
        self.assertEqual(
            header[2],
            " --- Energy the sphere adds, estimated from the free atoms: 2.8E-08 "
            "Ry (aim 5.0E-04 Ry; an estimate, not a bound" + left_out,
        )
        self.assertEqual(
            setup[1],
            " --- Energy the sphere adds, estimated from the free atoms: 2.8E-08 "
            "Ry (wall share of Domain_Energy_Tolerance 5.0E-04 Ry; an estimate, "
            "not a bound" + left_out,
        )
        # Every species has its density: nothing is left out, nothing said.
        self.assertNotIn("without", domain_header_lines(nanodiamond(radius=False))[2])

    def test_tail_on_a_plan_that_is_not_engaged_gets_no_number(self) -> None:
        # Hartree_Atomic_Tail: on, or its switch, adds the tail to PARSEC's
        # order where the estimate is within a tenth of the tolerance.  Those
        # are not PARSEC's values, and nothing was measured on them.
        text = parsed_inputs.BASE_INPUT
        for translation, last in (
            (
                parse_text(text + "Hartree_Atomic_Tail: on\n", BASE_PSEUDOPOTENTIALS),
                " NOTE: the boundary values of this run are outside the "
                "calibration of Domain_Energy_Tolerance, which then covers the "
                "wall only",
            ),
            (
                parse_text(
                    parsed_inputs.without_radius(text) + "Hartree_Atomic_Tail: on\n",
                    BASE_PSEUDOPOTENTIALS,
                ),
                " WARNING: the boundary values of this run are not the ones the "
                "default radius was chosen for; Domain_Energy_Tolerance covers "
                "the wall only",
            ),
        ):
            system = stand_in(translation)
            plan = system.hartree_boundary
            self.assertEqual((plan.engaged, plan.atomic_tail, plan.order), (False, True, 9))
            lines, record = domain_setup_lines(system, translation)
            self.assertEqual(
                lines[-2:],
                [
                    " --- Energy left by the boundary values: no estimate (atomic "
                    "tail on a plan that is not engaged)",
                    last,
                ],
            )
            self.assertEqual(
                record["boundary"],
                dict(
                    about_ry=None,
                    bound_ry=None,
                    plan="atomic tail on a plan that is not engaged",
                ),
            )
            self.assertFalse(any("PARSEC's values" in line for line in lines))
        # Without the tail the values are PARSEC's and keep their bound.
        plain = parse_text(text, BASE_PSEUDOPOTENTIALS)
        lines, record = domain_setup_lines(stand_in(plain), plain)
        self.assertEqual(record["boundary"]["plan"], "not engaged")
        self.assertIsNotNone(record["boundary"]["bound_ry"])
        self.assertTrue(any("PARSEC's values" in line for line in lines))

    def test_rule_without_a_radius_for_an_input_that_has_one(self) -> None:
        # A tolerance below the round-off of the estimate is a way to ask for
        # order 60, and ran before the rule existed.  The rule has no radius
        # for it; the input has, so the line says so and the set-up goes on.
        translation = parse_text(
            parsed_inputs.two_atom_smoke(
                "Hartree_Boundary_Tolerance: 1e-200 Ry", "Output_Level: 2"
            ),
            DATA,
        )
        lines, record = domain_setup_lines(stand_in(translation), translation)
        self.assertEqual(
            lines[-3:],
            [
                " --- Energy left by the boundary values: no estimate (largest "
                "multipole order reached)",
                " NOTE: the boundary values of this run are outside the "
                "calibration of Domain_Energy_Tolerance, which then covers the "
                "wall only",
                " --- Without Boundary_Sphere_Radius the default rule 1 would "
                "give no radius: no radius meets the Hartree boundary tolerance "
                "of 1.0e-200 Ry at the largest multipole order 60",
            ],
        )
        self.assertEqual(record["radius_from"], "input")

    def test_boundary_line_follows_the_plan_in_use(self) -> None:
        from parsec_python.acceleration.driver import _apply_hartree_boundary_switches

        chosen, given = nanodiamond(radius=False), nanodiamond(radius=True)
        warning = (
            " WARNING: the boundary values of this run are not the ones the "
            "default radius was chosen for; Domain_Energy_Tolerance covers the "
            "wall only"
        )
        note = (
            " NOTE: the boundary values of this run are outside the "
            "calibration of Domain_Energy_Tolerance, which then covers the "
            "wall only"
        )
        names = (
            "PARSEC_HARTREE_BOUNDARY",
            "PARSEC_HARTREE_BOUNDARY_TOLERANCE",
            "PARSEC_HARTREE_ATOMIC_TAIL",
            "PARSEC_HARTREE_LPOLE",
        )
        clean = {key: value for key, value in os.environ.items() if key not in names}
        for switch, value, expected in (
            ("PARSEC_HARTREE_BOUNDARY", "legacy", "no estimate (tolerance off)"),
            ("PARSEC_HARTREE_BOUNDARY_TOLERANCE", "off", "no estimate (tolerance off)"),
            ("PARSEC_HARTREE_ATOMIC_TAIL", "off", "no estimate (atomic tail off)"),
            (
                "PARSEC_HARTREE_BOUNDARY_TOLERANCE",
                "1e-9",
                "no estimate (largest multipole order reached)",
            ),
            (
                "PARSEC_HARTREE_BOUNDARY_TOLERANCE",
                "1e-5",
                "at order 32 (tolerance 1.0E-05 Ry, atomic tail): about 1.1E-06 "
                "Ry, calibrated bound 3.5E-06 Ry (aim 5.0E-04 Ry)",
            ),
            (
                "PARSEC_HARTREE_LPOLE",
                "60",
                "at order 60 (PARSEC's values, their estimate is within a tenth "
                "of the tolerance 1.0E-03 Ry): calibrated bound 3.6E-09 Ry (aim "
                "5.0E-04 Ry)",
            ),
        ):
            with self.subTest(switch=switch, value=value), patch.dict(
                os.environ, {**clean, switch: value}, clear=True
            ):
                lines, record = domain_setup_lines(
                    stand_in(chosen, _apply_hartree_boundary_switches(chosen.problem)),
                    chosen,
                )
                self.assertEqual(
                    lines[0],
                    " --- Energy left by the boundary values"
                    + (": " if expected.startswith("no") else " ")
                    + expected,
                )
                if expected.startswith("no"):
                    self.assertEqual(lines[1:], [warning])
                    self.assertIsNone(record["boundary"]["bound_ry"])
                    self.assertIsNone(record["boundary"]["about_ry"])
                    lines, _record = domain_setup_lines(
                        stand_in(given, _apply_hartree_boundary_switches(given.problem)),
                        given,
                    )
                    if "largest" not in expected:
                        self.assertEqual(lines[-1], note)
                else:
                    self.assertNotIn(warning, lines)
                    self.assertIsNotNone(record["boundary"]["bound_ry"])
        # The direct Coulomb sum leaves nothing.
        exact = replace(
            given.problem, hartree=replace(given.problem.hartree, boundary_method="direct")
        )
        lines, record = domain_setup_lines(stand_in(given, exact), given)
        self.assertEqual(lines[-1], " --- Energy left by the boundary values: none, they are exact")
        self.assertEqual(record["boundary"]["bound_ry"], 0.0)


class DensityAtTheSphereTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.potentials = load_pseudopotentials(
            {
                symbol: SpeciesPotential(
                    BASE_PSEUDOPOTENTIALS / f"{symbol}_POTRE.DAT", 0
                )
                for symbol in ("C", "H")
            }
        )

    @staticmethod
    def profile(spacing, radius, decay, slope):
        """A grid and a density whose shells are ``G sinh(kappa s)**2 / kappa**2``."""

        grid = build_cluster_grid(
            GridSettings(spacing=spacing * domain.ANGSTROM, radius=radius * domain.ANGSTROM)
        )
        distance = np.linalg.norm(grid.coordinates, axis=1)
        depth = grid.settings.radius - distance
        density = (
            slope
            * np.sinh(decay * depth) ** 2
            / decay**2
            / (4.0 * np.pi * np.maximum(distance, 1.0e-9) ** 2)
        )
        return grid, density

    def result(self, grid, density, *, converged=True, levels=(-0.6, -0.3, 0.05)):
        return SimpleNamespace(
            converged=converged,
            grid=grid,
            density=density,
            eigenvalues=np.array(levels),
            occupations=np.array([1.0, 1.0, 0.0][: len(levels)]),
            atoms=(Atom("C", (0.0, 0.0, 1.2)), Atom("H", (0.0, 0.0, -2.0))),
            pseudopotentials=self.potentials,
            electron_count=5.0,
        )

    def test_shell_sums_are_those_of_the_grid(self) -> None:
        grid, density = self.profile(0.2, 4.9, 0.9, 1.0e-3)
        sums = domain.sphere_shell_sums(
            grid.coordinates, density, grid.settings.radius, grid.volume_element
        )
        depth = grid.settings.radius - np.linalg.norm(grid.coordinates, axis=1)
        self.assertEqual(sums.shape, (10,))
        for shell in range(10):
            inside = (depth >= 0.4 * shell) & (depth < 0.4 * (shell + 1))
            self.assertAlmostEqual(
                sums[shell],
                density[inside].sum() * grid.volume_element / 0.4,
                delta=1.0e-12 * sums.max(),
            )
        # Blocks of rows change nothing.
        with patch.object(domain, "_SHELL_BLOCK_ROWS", 1000):
            np.testing.assert_allclose(
                domain.sphere_shell_sums(
                    grid.coordinates, density, grid.settings.radius, grid.volume_element
                ),
                sums,
                rtol=1.0e-12,
            )

    def test_fit_recovers_a_profile_on_fine_and_coarse_grids(self) -> None:
        for spacing in (0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
            for radius, decay, slope in ((4.9, 0.9, 1.0e-3), (7.0, 0.5, 2.0e-4), (7.0, 1.4, 5.0e-5)):
                grid, density = self.profile(spacing, radius, decay, slope)
                fit = domain.wall_from_density(
                    grid.coordinates, density, grid.settings.radius, grid.volume_element
                )
                with self.subTest(spacing=spacing, radius=radius, decay=decay):
                    limit = 0.05 if spacing <= 0.25 else 0.15
                    self.assertAlmostEqual(
                        fit.energy / (slope / (2.0 * decay)), 1.0, delta=limit
                    )
                    self.assertAlmostEqual(fit.decay, decay, delta=0.03)
                    self.assertEqual((fit.shells, fit.rough), (8, False))
                    self.assertLess(fit.rms, 0.15)

    def test_fit_of_the_stored_shells_of_the_measured_runs(self) -> None:
        self.assertEqual(len(SHELLS), 83)
        ratios = []
        for run in SHELLS:
            fit = domain.fit_wall(np.array(run["shell_charge_per_bohr"]))
            reference = run["reference_fit"]
            self.assertAlmostEqual(fit.energy / reference["energy_ry"], 1.0, delta=1.0e-12)
            self.assertAlmostEqual(fit.decay, reference["decay_per_bohr"], delta=1.0e-12)
            self.assertAlmostEqual(fit.rms, reference["rms"], delta=1.0e-12)
            self.assertEqual(fit.at_scan_end, reference["at_scan_end"])
            if (
                not run["widest"]
                and run["vacuum_angstrom"] >= 3.0
                and run["energy_minus_widest_ry"] > 3.0e-6
            ):
                ratios.append(
                    (
                        run["energy_minus_widest_ry"] / fit.energy,
                        run["system"],
                        run["vacuum_angstrom"],
                        fit.rms,
                    )
                )
        # Measured energy above the widest sphere over the estimate: the
        # accuracy stated for the rule.
        values = np.array([row[0] for row in ratios])
        self.assertEqual(len(values), 43)
        self.assertEqual(min(ratios)[1:3], ("mg2", 3.0))
        self.assertAlmostEqual(values.min(), 0.306, delta=1.0e-3)
        self.assertAlmostEqual(values.max(), 0.933, delta=1.0e-3)
        self.assertAlmostEqual(float(np.median(values)), 0.827, delta=1.0e-3)
        self.assertLess(max(row[3] for row in ratios), 0.097)
        wide = np.array([row[0] for row in ratios if row[2] >= 3.5])
        self.assertEqual(len(wide), 32)
        self.assertAlmostEqual(wide.min(), 0.549, delta=1.0e-3)
        settled = np.array(
            [
                row[0]
                for row in ratios
                if row[2] >= 3.5 and not (row[1] in ("na2", "mg2") and row[2] < 5.0)
            ]
        )
        self.assertEqual(len(settled), 30)
        self.assertGreater(settled.min(), 0.69)

    def test_radius_that_would_meet_the_wall_share(self) -> None:
        # The benchmark cluster from its four stored runs: 3.8 to 3.9 ang of
        # vacuum, with the tolerance of a vacuum below 4 ang.
        system = stand_in(nanodiamond(radius=True))
        found = {}
        for run in SHELLS:
            if run["system"] != "k06":
                continue
            fit = domain.fit_wall(np.array(run["shell_charge_per_bohr"]))
            # The series bound spares this cluster every estimate.
            with patch.object(
                domain, "estimate_omitted_potential", side_effect=AssertionError
            ):
                wanted = domain.radius_for_budget(
                    fit,
                    run["radius_bohr"],
                    system.atoms,
                    system.pseudopotentials,
                    system.electron_count,
                    spacing=system.input.grid.spacing,
                    highest_occupied=run["highest_occupied_ry"],
                )
            self.assertEqual(
                (wanted.estimates, wanted.wall_radius_text), (0, wanted.radius_text)
            )
            found[run["vacuum_angstrom"]] = (
                wanted.radius_text,
                wanted.tolerance_text,
                wanted.limit,
                wanted.set_by,
            )
            if run["vacuum_angstrom"] == 4.0:
                self.assertEqual(f"{fit.energy:.1E} {fit.decay:.2f}", "1.7E-04 0.82")
        self.assertEqual(
            found,
            {
                3.0: ("16.1 ang", "6.7e-04 Ry", "", "density at the sphere"),
                3.5: ("16.1 ang", "6.7e-04 Ry", "", "density at the sphere"),
                4.0: ("16.1 ang", "6.7e-04 Ry", "", "density at the sphere"),
                5.0: ("16.2 ang", "6.7e-04 Ry", "", "density at the sphere"),
            },
        )

    def test_step_is_limited_and_follows_the_highest_occupied_level(self) -> None:
        fit = domain.WallFit(
            prefactor=0.03, decay=0.15, energy=0.1, rms=0.19, at_scan_end=True, shells=8
        )
        self.assertTrue(fit.rough)
        atoms = (Atom("H", (0.0, 0.0, 2.9)), Atom("H", (0.0, 0.0, -2.9)))
        options = dict(spacing=0.4, highest_occupied=-0.2)
        radius = 2.9 + 3.0 * domain.ANGSTROM
        # A thin sphere: kappa at the floor of its scan asks for 10 ang
        # more; the level asks for 3.4 ang, and the limit allows 2.5 ang.
        wanted = domain.radius_for_budget(fit, radius, atoms, self.potentials, 2.0, **options)
        self.assertEqual(wanted.limit, "at least")
        self.assertAlmostEqual(wanted.decay, math.sqrt(0.2))
        self.assertLessEqual(wanted.radius - radius, 2.6 * domain.ANGSTROM)
        self.assertGreater(wanted.radius - radius, 2.5 * domain.ANGSTROM - 1.0e-9)
        # A level above zero is no decay constant.
        unbound = domain.radius_for_budget(
            fit, radius, atoms, self.potentials, 2.0, spacing=0.4, highest_occupied=0.1
        )
        self.assertEqual((unbound.decay, unbound.limit), (0.15, "at least"))
        # A wide sphere comes in by at most the limit, and never below the
        # minimum vacuum.
        small = replace(fit, energy=1.0e-12, decay=0.9, rms=0.02, at_scan_end=False)
        wide = 2.9 + 8.0 * domain.ANGSTROM
        wanted = domain.radius_for_budget(small, wide, atoms, self.potentials, 2.0, **options)
        self.assertEqual(wanted.limit, "limit")
        self.assertAlmostEqual((wide - wanted.radius) / domain.ANGSTROM, 2.5, delta=0.1)
        wanted = domain.radius_for_budget(
            small, 2.9 + 3.4 * domain.ANGSTROM, atoms, self.potentials, 2.0, **options
        )
        self.assertEqual(wanted.set_by, "minimum vacuum")
        self.assertAlmostEqual((wanted.radius - 2.9) / domain.ANGSTROM, 3.0, delta=0.1)
        # The vacuum held the radius there, not the limit of the step: also
        # from 5 ang, where the step of 2.5 ang would end below it.
        self.assertEqual(wanted.limit, "")
        wanted = domain.radius_for_budget(
            small, 2.9 + 5.0 * domain.ANGSTROM, atoms, self.potentials, 2.0, **options
        )
        self.assertEqual(
            (wanted.set_by, wanted.limit, wanted.wall_radius_text),
            ("minimum vacuum", "", wanted.radius_text),
        )
        # A tolerance of the input stays the input's, and the direct Coulomb
        # sum has none.
        self.assertIsNone(
            domain.radius_for_budget(
                small, wide, atoms, self.potentials, 2.0, boundary_tolerance=1.0e-4, **options
            ).tolerance_text
        )
        self.assertIsNone(
            domain.radius_for_budget(
                small, wide, atoms, self.potentials, 2.0, multipole_boundary=False, **options
            ).boundary_tolerance
        )

    def test_radius_keeps_the_stencil_of_the_rule(self) -> None:
        # Seven points of 1.2 bohr beyond the outermost atom are more than
        # 3 ang: the rule never gives less, nor does the radius after the SCF.
        atoms = (Atom("H", (0.0, 0.0, 2.9)), Atom("H", (0.0, 0.0, -2.9)))
        small = domain.WallFit(
            prefactor=1.8e-12, decay=0.9, energy=1.0e-12, rms=0.02, at_scan_end=False, shells=8
        )
        rule = domain.default_domain(atoms, self.potentials, 2.0, spacing=1.2, budget=1.0)
        self.assertEqual((rule.radius_text, rule.set_by), ("6.0 ang", "stencil"))
        # A step in of 2.5 ang from 6.5 and from 5 ang of vacuum: to 4 ang of
        # it, and to the minimum of 3 ang.
        for vacuum in (6.5, 5.0):
            wanted = domain.radius_for_budget(
                small, 2.9 + vacuum * domain.ANGSTROM, atoms, self.potentials, 2.0,
                spacing=1.2, highest_occupied=-0.2,
            )
            self.assertEqual(
                (wanted.radius_text, wanted.set_by, wanted.limit),
                (rule.radius_text, "stencil", ""),
            )
            self.assertGreaterEqual(wanted.radius, 2.9 + 7 * 1.2)
        # A stencil of two points a side leaves the minimum vacuum in charge.
        wanted = domain.radius_for_budget(
            small, 2.9 + 5.0 * domain.ANGSTROM, atoms, self.potentials, 2.0,
            spacing=1.2, stencil_half_width=2, highest_occupied=-0.2,
        )
        self.assertEqual((wanted.radius_text, wanted.set_by), ("4.6 ang", "minimum vacuum"))

    def test_largest_order_is_asked_for_the_tolerance_of_the_rule_alone(self) -> None:
        # One hydrogen atom 100 ang from the centre, whose radius the rule
        # sets by the multipole order.  From that sphere the wall would
        # allow a step in; order 60 does not, and the radius says so.
        distance = 100.0 * domain.ANGSTROM
        atom = [Atom("H", (0.0, 0.0, distance))]
        rule = domain.default_domain(atom, self.potentials, 1.0, spacing=0.4)
        self.assertEqual(rule.set_by, "multipole order")
        small = domain.WallFit(
            prefactor=1.8e-12, decay=0.9, energy=1.0e-12, rms=0.02, at_scan_end=False, shells=8
        )
        options = dict(spacing=0.4, highest_occupied=-0.5)
        wanted = domain.radius_for_budget(
            small, rule.radius, atom, self.potentials, 1.0, **options
        )
        self.assertEqual(
            (wanted.radius_text, wanted.tolerance_text, wanted.set_by, wanted.limit),
            (rule.radius_text, rule.tolerance_text, "multipole order", "limit"),
        )
        self.assertGreater(wanted.estimates, 0)
        self.assertAlmostEqual(
            float(rule.radius_text.split()[0]) - float(wanted.wall_radius_text.split()[0]),
            2.5,
            delta=0.1,
        )
        # A tolerance of the input is not the rule's to meet: the radius is
        # the wall's, and nothing is estimated.
        with patch.object(
            domain, "estimate_omitted_potential", side_effect=AssertionError
        ):
            given = domain.radius_for_budget(
                small, rule.radius, atom, self.potentials, 1.0,
                boundary_tolerance=1.0e-3, **options,
            )
        self.assertEqual(
            (given.radius_text, given.set_by, given.estimates, given.tolerance_text),
            (wanted.wall_radius_text, "density at the sphere", 0, None),
        )

    def test_lines_say_what_set_the_radius(self) -> None:
        translation = SimpleNamespace(
            domain_energy_tolerance=1.0e-3, boundary_tolerance_from="default"
        )
        # A wall energy far below its share, from three spheres: the minimum
        # vacuum, the stencil of a coarse grid and the limit of the step hold
        # the radius in turn.
        for spacing, radius, line, set_by, limit in (
            (
                0.25,
                5.2,
                "       Boundary_Sphere_Radius: 4.1 ang   # the minimum vacuum of 3 ang",
                "minimum vacuum",
                "",
            ),
            (
                0.5,
                7.0,
                "       Boundary_Sphere_Radius: 4.6 ang   # the stencil of the grid "
                "needs this much",
                "stencil",
                "",
            ),
            (
                0.3,
                7.2,
                "       Boundary_Sphere_Radius: 4.7 ang   # the step is limited to "
                "2.5 ang; a smaller sphere may do",
                "density at the sphere",
                "limit",
            ),
        ):
            grid, density = self.profile(spacing, radius, 0.9, 1.0e-10)
            lines, record = domain_finish_lines(
                self.result(grid, density), translation, HartreeSettings()
            )
            with self.subTest(radius=radius):
                self.assertEqual(lines[3], line)
                self.assertEqual(
                    (record["radius_set_by"], record["radius_limit"], record["estimates"]),
                    (set_by, limit, 0),
                )
                self.assertEqual(
                    record["radius_of_the_wall_alone"], record["boundary_sphere_radius"]
                )
        # The seconds of the shells and of the radius are kept apart.
        self.assertGreater(record["shell_seconds"], 0.0)
        self.assertGreaterEqual(record["radius_seconds"], 0.0)
        self.assertGreaterEqual(
            record["seconds"], record["shell_seconds"] + record["radius_seconds"]
        )
        # A radius the largest order holds, as for 39,368 electrons below
        # 4 ang of vacuum, and one moved off the grid points.
        grid, density = self.profile(0.25, 5.2, 0.9, 5.0e-4)
        held = domain.RadiusForBudget(
            radius_text="34.2 ang",
            radius=34.2 * domain.ANGSTROM,
            tolerance_text="2.0e-04 Ry",
            boundary_tolerance=2.0e-4,
            set_by="multipole order",
            limit="",
            decay=0.85,
            estimates=5,
            wall_radius_text="33.9 ang",
        )
        with patch.object(parsec_output, "radius_for_budget", return_value=held):
            lines, record = domain_finish_lines(
                self.result(grid, density), translation, HartreeSettings()
            )
        self.assertEqual(
            lines[2:5],
            [
                "   For this system the wall share would be met by:",
                "       Boundary_Sphere_Radius: 34.2 ang   # order 60 misses the "
                "tolerance below this radius; the wall alone would allow 33.9 ang",
                "       Hartree_Boundary_Tolerance: 2.0e-04 Ry",
            ],
        )
        self.assertEqual(
            (record["radius_set_by"], record["radius_of_the_wall_alone"], record["estimates"]),
            ("multipole order", "33.9 ang", 5),
        )
        # With their comment the two lines are still an input.
        pasted = parse_text(
            parsed_inputs.without_radius(parsed_inputs.BASE_INPUT)
            + "\n".join(lines[3:5])
            + "\n",
            BASE_PSEUDOPOTENTIALS,
        )
        self.assertEqual(pasted.problem.grid.radius, held.radius)
        self.assertEqual(pasted.problem.hartree.boundary_tolerance, 2.0e-4)
        moved = replace(
            held,
            radius_text="4.95 ang",
            set_by="density at the sphere (+0.05 ang: a grid point lay on the sphere)",
            limit="at least",
        )
        with patch.object(parsec_output, "radius_for_budget", return_value=moved):
            lines, _record = domain_finish_lines(
                self.result(grid, density), translation, HartreeSettings()
            )
        self.assertEqual(
            lines[3],
            "       Boundary_Sphere_Radius: 4.95 ang   # at least: the step is "
            "limited to 2.5 ang; +0.05 ang: a grid point lay on the sphere",
        )

    @needs_measurements
    def test_radius_after_the_scf_of_the_largest_benchmark_cluster(self) -> None:
        # C9449H1572 with the 35.1 ang of its input.  Down to 4 ang of vacuum
        # one estimate tells that order 60 meets the tolerance of the rule;
        # below, the tolerance halves, order 60 misses it and the radius
        # stays at 34.2 ang, at the price of a search.
        folder = Path(MEASUREMENTS) / "recipe_inputs" / "k14_ff4_fd30_d15" / "python_sad"
        text, folder = source_of(folder / "parsec.in", None)
        calls = []

        def counted(*arguments, **options):
            calls.append(arguments[2])
            return estimate_omitted_potential(*arguments, **options)

        def after(translation, energy):
            del calls[:]
            given = translation.boundary_tolerance_from == "input"
            wanted = domain.radius_for_budget(
                domain.WallFit(
                    prefactor=1.7 * energy, decay=0.85, energy=energy,
                    rms=0.05, at_scan_end=False, shells=8,
                ),
                translation.problem.grid.radius,
                system.atoms,
                system.pseudopotentials,
                system.electron_count,
                spacing=translation.problem.grid.spacing,
                highest_occupied=-0.45,
                boundary_tolerance=(
                    translation.problem.hartree.boundary_tolerance if given else None
                ),
            )
            self.assertEqual(wanted.estimates, len(calls))
            return (
                wanted.radius_text,
                wanted.tolerance_text,
                wanted.set_by,
                wanted.wall_radius_text,
                len(calls),
            )

        translation = parse_text(text, folder)
        system = stand_in(translation)
        with patch.object(domain, "estimate_omitted_potential", counted):
            self.assertEqual(
                after(translation, 1.0e-5),
                ("34.1 ang", "4.0e-04 Ry", "density at the sphere", "34.1 ang", 1),
            )
            found = after(translation, 1.0e-6)
            self.assertEqual(
                found[:4], ("34.2 ang", "2.0e-04 Ry", "multipole order", "33.3 ang")
            )
            self.assertLessEqual(found[4], 7)
            # With a tolerance line in the input nothing is estimated,
            # whatever the tolerance: 3e-5 Ry grew the sphere to 35.2 ang in
            # five to nine estimates, and 1e-99 Ry raised after nine.
            for tolerance in ("3e-5", "1e-99"):
                tight = parse_text(
                    text + f"Hartree_Boundary_Tolerance: {tolerance} Ry\n", folder
                )
                self.assertEqual(
                    after(tight, 1.0e-6),
                    ("33.3 ang", None, "density at the sphere", "33.3 ang", 0),
                )

    def test_lines_after_a_converged_scf(self) -> None:
        grid, density = self.profile(0.25, 5.2, 0.9, 5.0e-4)
        translation = SimpleNamespace(domain_energy_tolerance=1.0e-3, boundary_tolerance_from="default")
        lines, record = domain_finish_lines(
            self.result(grid, density), translation, HartreeSettings()
        )
        self.assertEqual(
            lines,
            [
                " Density at the sphere: the sphere adds an estimated 2.6E-04 Ry "
                "to the total energy (decay constant 0.92 /bohr: a factor 10 "
                "per 0.67 ang)",
                "   Wall share of Domain_Energy_Tolerance (5.0E-04 Ry): within it.",
                "   For this system the wall share would be met by:",
                "       Boundary_Sphere_Radius: 5.2 ang",
                "       Hartree_Boundary_Tolerance: 1.0e-03 Ry",
                "   NOTE: lowest empty level +0.0500 Ry; the rule does not cover "
                "empty levels, which need more vacuum.",
            ],
        )
        self.assertEqual(
            (record["status"], record["rough"], record["boundary_sphere_radius"]),
            ("estimated", False, "5.2 ang"),
        )
        self.assertEqual(len(record["shell_charge_per_bohr"]), 10)
        # G / (2 kappa) of the profile, within the 6 % of this grid.
        self.assertAlmostEqual(record["energy_ry"], 5.0e-4 / 1.8, delta=1.7e-5)
        self.assertEqual((record["highest_occupied_ry"], record["lowest_empty_ry"]), (-0.3, 0.05))
        # The two lines are an input.
        radius = _physical_length(lines[3].split(":")[1], label="radius")
        self.assertEqual(radius, record["radius_bohr"])

    def test_notes_and_warnings_after_the_scf(self) -> None:
        translation = SimpleNamespace(domain_energy_tolerance=1.0e-3, boundary_tolerance_from="input")
        hartree = HartreeSettings(boundary_tolerance=1.0e-4)
        # Above the wall share: a note.  Above the whole budget: a warning
        # that names the radius.  The tolerance of the input is not repeated.
        grid, density = self.profile(0.25, 5.2, 0.9, 1.5e-3)
        lines, record = domain_finish_lines(self.result(grid, density), translation, hartree)
        self.assertEqual(
            lines[1],
            "   NOTE: this exceeds the wall share of Domain_Energy_Tolerance (5.0E-04 Ry).",
        )
        self.assertEqual(lines[3:4], ["       Boundary_Sphere_Radius: 5.5 ang"])
        self.assertIsNone(record["hartree_boundary_tolerance"])
        self.assertNotIn("Hartree_Boundary_Tolerance", "\n".join(lines))
        grid, density = self.profile(0.25, 5.2, 0.9, 2.0e-2)
        lines, record = domain_finish_lines(self.result(grid, density), translation, hartree)
        self.assertEqual(
            lines[1],
            "   WARNING: this exceeds Domain_Energy_Tolerance (1.0E-03 Ry); the "
            "wall share needs a radius of 6.2 ang.",
        )
        # The highest occupied level above zero, as in a small anion.
        lines, record = domain_finish_lines(
            self.result(grid, density, levels=(-0.2, 0.104, 0.3)), translation, hartree
        )
        self.assertIn(
            "   NOTE: the highest occupied level, +0.1040 Ry, is above zero: it "
            "is held by the sphere; the energy has no limit for a large sphere.",
            lines,
        )
        # No empty level, no note of one.
        lines, record = domain_finish_lines(
            self.result(grid, density, levels=(-0.6, -0.3)), translation, hartree
        )
        self.assertFalse(any("lowest empty level" in line for line in lines))
        self.assertIsNone(record["lowest_empty_ry"])

    def test_tolerance_of_the_input_asks_for_no_estimate(self) -> None:
        # The radius after the SCF is the wall's where the input gives the
        # tolerance: no order is asked for it, whatever it is.  The search
        # of the rule made nine estimates for the last of these and raised.
        grid, density = self.profile(0.25, 5.2, 0.9, 5.0e-4)
        translation = SimpleNamespace(
            domain_energy_tolerance=1.0e-3, boundary_tolerance_from="input"
        )
        with patch.object(
            domain, "estimate_omitted_potential", side_effect=AssertionError
        ):
            for tolerance in (1.0e-4, 1.0e-30, 1.0e-200):
                lines, record = domain_finish_lines(
                    self.result(grid, density),
                    translation,
                    HartreeSettings(boundary_tolerance=tolerance),
                )
                self.assertEqual(
                    lines[1:4],
                    [
                        "   Wall share of Domain_Energy_Tolerance (5.0E-04 Ry): "
                        "within it.",
                        "   For this system the wall share would be met by:",
                        "       Boundary_Sphere_Radius: 5.2 ang",
                    ],
                )
                self.assertEqual(
                    (record["estimates"], record["radius_set_by"]),
                    (0, "density at the sphere"),
                )

    def test_estimate_stands_where_no_radius_can_be_given(self) -> None:
        # A tolerance of the rule that no order meets at any radius: the
        # search gives up, and the block keeps what it knows.
        grid, density = self.profile(0.25, 5.2, 0.9, 5.0e-4)
        translation = SimpleNamespace(
            domain_energy_tolerance=1.0e-300, boundary_tolerance_from="default"
        )
        lines, record = domain_finish_lines(
            self.result(grid, density), translation, HartreeSettings()
        )
        self.assertEqual(
            lines[:2],
            [
                " Density at the sphere: the sphere adds an estimated 2.6E-04 Ry "
                "to the total energy (decay constant 0.92 /bohr: a factor 10 "
                "per 0.67 ang)",
                "   WARNING: this exceeds Domain_Energy_Tolerance (1.0E-300 Ry).",
            ],
        )
        self.assertRegex(
            lines[2],
            r"^   No radius for the wall share: no radius meets the Hartree "
            r"boundary tolerance of \d\.\de-299 Ry at the largest multipole "
            r"order 60\.$",
        )
        self.assertEqual(
            lines[3:],
            [
                "   NOTE: lowest empty level +0.0500 Ry; the rule does not cover "
                "empty levels, which need more vacuum.",
            ],
        )
        self.assertEqual(
            (record["status"], record["boundary_sphere_radius"], record["radius_bohr"]),
            ("estimated", None, None),
        )
        self.assertRegex(record["radius_set_by"], r"^none: no radius meets ")
        self.assertAlmostEqual(record["energy_ry"], 5.0e-4 / 1.8, delta=1.7e-5)
        json.dumps(record, allow_nan=False)

    def test_rough_fits_are_printed_and_flagged(self) -> None:
        # A decay slower than the scan: the fit ends at 0.15 /bohr.
        grid, density = self.profile(0.3, 7.0, 0.05, 2.0e-3)
        translation = SimpleNamespace(domain_energy_tolerance=1.0e-3, boundary_tolerance_from="default")
        lines, record = domain_finish_lines(
            self.result(grid, density, levels=(0.02, 0.1, 0.3)), translation, HartreeSettings()
        )
        self.assertTrue(record["rough"] and record["decay_at_scan_end"])
        self.assertEqual(record["decay_per_bohr"], 0.15)
        self.assertEqual(
            lines[1], "   This estimate is rough: the decay constant is at an end of its scan."
        )
        self.assertRegex(lines[2], r"^   WARNING: this exceeds Domain_Energy_Tolerance")
        self.assertRegex(lines[2], r"ang at least \(the step is limited to 2\.5 ang\)\.$")
        self.assertRegex(lines[4], r"# at least: the step is limited to 2\.5 ang$")
        self.assertEqual(record["radius_limit"], "at least")

    def test_lines_for_a_grid_whose_lengths_are_numpy_scalars(self) -> None:
        # The grid of a caller that built its settings from NumPy scalars.
        plain_grid, density = self.profile(0.25, 5.2, 0.9, 5.0e-4)
        grid = build_cluster_grid(
            GridSettings(
                spacing=np.float64(plain_grid.settings.spacing),
                radius=np.float64(plain_grid.settings.radius),
            )
        )
        self.assertIsInstance(grid.settings.radius, np.float64)
        translation = SimpleNamespace(
            domain_energy_tolerance=1.0e-3, boundary_tolerance_from="default"
        )
        lines, record = domain_finish_lines(
            self.result(grid, density), translation, HartreeSettings()
        )
        expected, plain = domain_finish_lines(
            self.result(plain_grid, density), translation, HartreeSettings()
        )
        self.assertEqual(lines, expected)
        self.assertEqual(lines[3], "       Boundary_Sphere_Radius: 5.2 ang")
        self.assertEqual(record["radius_bohr"], plain["radius_bohr"])
        self.assertIs(type(record["radius_bohr"]), float)

    def test_no_estimate_without_convergence_or_shells(self) -> None:
        grid, density = self.profile(0.3, 4.9, 0.9, 1.0e-3)
        translation = SimpleNamespace()
        self.assertEqual(
            domain_finish_lines(self.result(grid, density, converged=False), translation),
            (
                [" Density at the sphere: no estimate, the SCF did not converge"],
                dict(status="not converged"),
            ),
        )
        lines, record = domain_finish_lines(
            self.result(grid, np.zeros(grid.size)), translation
        )
        self.assertEqual(
            lines,
            [
                " Density at the sphere: no estimate, fewer than three shells "
                "inside the sphere hold charge"
            ],
        )
        self.assertEqual(record["status"], "no fit")
        box = SimpleNamespace(
            converged=True, grid=SimpleNamespace(settings=SimpleNamespace(domain_shape="box"))
        )
        self.assertEqual(domain_finish_lines(box, translation), ([], None))
        self.assertEqual(domain_finish_lines(SimpleNamespace(converged=True), translation), ([], None))


@contextmanager
def domain_report(value: str | None):
    """The environment with ``PARSEC_DOMAIN_REPORT`` at ``value``, or without it."""

    with patch.dict(os.environ):
        os.environ.pop("PARSEC_DOMAIN_REPORT", None)
        if value is not None:
            os.environ["PARSEC_DOMAIN_REPORT"] = value
        yield


class CommandLineTests(unittest.TestCase):
    def run_smoke(self, text, *arguments, report=None):
        with tempfile.TemporaryDirectory() as folder, domain_report(report):
            folder = Path(folder).resolve()
            (folder / "parsec.in").write_bytes(text.encode("utf-8"))
            (folder / "H_POTRE.DAT").write_bytes((DATA / "H_POTRE.DAT").read_bytes())
            output = StringIO()
            with redirect_stdout(output):
                code = cli_main([str(folder / "parsec.in"), "--no-archive", *arguments])
            log = folder / "parsec.out"
            return code, output.getvalue(), log.read_text(encoding="utf-8") if log.exists() else ""

    def test_switch_restores_the_report_of_an_input_with_a_radius(self) -> None:
        for value, expected in (
            (None, True),
            ("1", True),
            ("on", True),
            ("", True),
            ("0", False),
            (" OFF ", False),
            ("false", False),
            ("no", False),
        ):
            with domain_report(value):
                self.assertIs(domain_report_requested(), expected, value)
        # One converged calculation with the report, keeping the lines its
        # two estimates return, and one without: the second log is the first
        # with exactly those lines taken out.  Digits are masked, for the
        # times, and the folder of a run is left out.
        text = parsed_inputs.two_atom_smoke()
        made = {}

        def kept(name):
            estimate = getattr(parsec_output, name)

            def keeping(*arguments):
                made[name] = estimate(*arguments)[0]
                return estimate(*arguments)

            return patch.object(parsec_output, name, keeping)

        with kept("domain_setup_lines"), kept("domain_finish_lines"):
            code, _console, with_report = self.run_smoke(text, "--quiet")
        self.assertEqual(code, 0)
        self.assertEqual(len(made["domain_setup_lines"]), 4)
        self.assertRegex(made["domain_finish_lines"][0], r"^ Density at the sphere: the")
        with (
            patch.object(parsec_output, "domain_setup_lines", side_effect=AssertionError),
            patch.object(parsec_output, "domain_finish_lines", side_effect=AssertionError),
        ):
            code, _console, without = self.run_smoke(text, "--quiet", report="0")
        self.assertEqual(code, 0)
        for words in (
            "Sphere of the input",
            "Energy the sphere adds",
            "Energy left by the boundary values",
            "Density at the sphere",
            "wall share",
        ):
            self.assertIn(words, with_report)
            self.assertNotIn(words, without)
        self.assertNotIn("the report failed", without)

        def masked(lines):
            files = ("parsec.in", "parsec.out", "H_POTRE.DAT")
            return [
                re.sub(
                    r"\d",
                    "#",
                    line.split(":")[0] if any(name in line for name in files) else line,
                )
                for line in lines
            ]

        expected = masked(with_report.split("\n"))
        for block in (made["domain_setup_lines"], ["", *made["domain_finish_lines"]]):
            block = masked(block)
            found = [
                start
                for start in range(len(expected) - len(block) + 1)
                if expected[start : start + len(block)] == block
            ]
            self.assertEqual(len(found), 1, block)
            del expected[found[0] : found[0] + len(block)]
        self.assertEqual(masked(without.split("\n")), expected)
        # Its dry run says nothing of the rule either.
        code, console, _log = self.run_smoke(text, "--dry-run", report="0")
        self.assertEqual(code, 0)
        self.assertNotIn("default rule", console)
        self.assertIn("Dry run successful", console)

    def test_calculation_with_a_radius_estimates_the_boundary_once(self) -> None:
        # From the parser to the last line of the report: the estimate of
        # the plan, at the radius of the input, and no other.  The radius
        # after the SCF is settled by the series bound.
        calls = []

        def counted(*arguments, **options):
            calls.append(arguments[2])
            return estimate_omitted_potential(*arguments, **options)

        with (
            patch.object(boundary_module, "estimate_omitted_potential", counted),
            patch.object(domain, "estimate_omitted_potential", counted),
        ):
            code, _console, log = self.run_smoke(parsed_inputs.two_atom_smoke(), "--quiet")
        self.assertEqual(code, 0)
        self.assertIn(
            "   For this system the wall share would be met by:", log.split("\n")
        )
        self.assertEqual(calls, [5.0])

    def test_switch_keeps_the_lines_of_a_radius_of_the_rule(self) -> None:
        # No former report to restore: the choice of the rule and what its
        # boundary values leave stay; the block after the SCF, which is the
        # part that costs, goes.
        text = parsed_inputs.without_radius(parsed_inputs.two_atom_smoke())
        text = text.replace("Grid_Spacing 0.8 bohr", "Grid_Spacing 1.2 bohr")
        with patch.object(parsec_output, "domain_finish_lines", side_effect=AssertionError):
            code, _console, log = self.run_smoke(text, "--quiet", report="0")
        self.assertEqual(code, 0)
        lines = log.split("\n")
        self.assertIn(
            " --- Radius 7.9 ang chosen by the default rule 1 "
            "(Domain_Energy_Tolerance = 1.0E-03 Ry)",
            lines,
        )
        self.assertIn("       Boundary_Sphere_Radius: 7.9 ang", lines)
        self.assertTrue(
            any(line.startswith(" --- Energy left by the boundary values") for line in lines)
        )
        self.assertIn("Self-consistency convergence achieved.", lines)
        self.assertNotIn("Density at the sphere", log)
        self.assertNotIn("the report failed", log)
        code, console, _log = self.run_smoke(text, "--dry-run", report="0")
        self.assertEqual(code, 0)
        self.assertIn(" --- Radius 7.9 ang chosen by the default rule 1 ", console)

    def test_scf_with_the_radius_left_to_the_rule(self) -> None:
        text = parsed_inputs.without_radius(
            (DATA / "H_cli_smoke.in").read_text(encoding="utf-8")
        )
        text = text.replace("Grid_Spacing 0.8 bohr", "Grid_Spacing 1.2 bohr")
        text = text.replace("Max_Iter: 1", "Max_Iter: 40")
        code, _console, log = self.run_smoke(text, "--quiet")
        self.assertEqual(code, 0)
        lines = log.split("\n")
        # The hydrogen file of the tests has no complete density: 7.5 ang.
        start = lines.index(" --- Radius is  14.172952 bohrs")
        self.assertEqual(
            lines[start + 1],
            " --- Radius 7.5 ang chosen by the default rule 1 "
            "(Domain_Energy_Tolerance = 1.0E-03 Ry)",
        )
        self.assertIn(" --- Radius set by: no atomic density for H", lines)
        self.assertIn("       Boundary_Sphere_Radius: 7.5 ang", lines)
        self.assertIn("       Hartree_Boundary_Tolerance: 1.0e-03 Ry", lines)
        self.assertTrue(
            any(line.startswith(" --- Energy left by the boundary values at order 9") for line in lines)
        )
        self.assertIn("Self-consistency convergence achieved.", lines)
        after = [index for index, line in enumerate(lines) if line.startswith(" Density at the sphere:")]
        self.assertEqual(len(after), 1)
        self.assertRegex(
            lines[after[0]],
            r"^ Density at the sphere: the sphere adds an estimated \d\.\dE-\d\d Ry "
            r"to the total energy \(decay constant \d\.\d\d /bohr: a factor 10 "
            r"per \d\.\d\d ang\)$",
        )
        self.assertEqual(
            lines[after[0] + 1],
            "   Wall share of Domain_Energy_Tolerance (5.0E-04 Ry): within it.",
        )
        self.assertEqual(lines[after[0] + 2], "   For this system the wall share would be met by:")
        self.assertRegex(lines[after[0] + 3], r"^       Boundary_Sphere_Radius: \d\.\d ang")

    def test_scf_that_does_not_converge_gets_no_estimate(self) -> None:
        text = (DATA / "H_cli_smoke.in").read_text(encoding="utf-8")
        code, _console, log = self.run_smoke(text, "--quiet")
        self.assertEqual(code, 3)
        self.assertIn(" Density at the sphere: no estimate, the SCF did not converge", log.split("\n"))
        self.assertIn(
            " --- Sphere of the input: outermost atom H at 0.000 ang from the "
            "centre; vacuum beyond it 2.117 ang",
            log.split("\n"),
        )

    def test_report_without_a_radius_does_not_stop_a_calculation(self) -> None:
        # The dry run of an input with a radius and a tolerance no order
        # meets ended with exit code 0 before the rule existed.
        text = parsed_inputs.two_atom_smoke("Hartree_Boundary_Tolerance: 1e-200 Ry")
        code, console, log = self.run_smoke(text, "--dry-run")
        self.assertEqual((code, log), (0, ""))
        self.assertIn(
            " --- Without Boundary_Sphere_Radius the default rule 1 would give "
            "no radius: no radius meets the Hartree boundary tolerance of "
            "1.0e-200 Ry at the largest multipole order 60\n",
            console,
        )
        self.assertIn("Dry run successful", console)
        # A complete calculation whose reports before and after the SCF find
        # no radius: it converges, says so twice and ends as it should.
        text = parsed_inputs.two_atom_smoke(
            "Domain_Energy_Tolerance: 1e-300 Ry", "Output_Level: 2"
        )
        code, _console, log = self.run_smoke(text, "--quiet")
        self.assertEqual(code, 0)
        lines = log.split("\n")
        self.assertIn("Self-consistency convergence achieved.", lines)
        self.assertEqual(
            sum(" would give no radius: no radius meets " in line for line in lines), 1
        )
        after = lines.index(
            "   WARNING: this exceeds Domain_Energy_Tolerance (1.0E-300 Ry)."
        )
        self.assertRegex(lines[after - 1], r"^ Density at the sphere: the sphere adds")
        self.assertRegex(
            lines[after + 1],
            r"^   No radius for the wall share: no radius meets the Hartree "
            r"boundary tolerance of \d\.\de-299 Ry at the largest multipole "
            r"order 60\.$",
        )
        self.assertFalse(any("Calculation failed" in line for line in lines))

    def test_report_that_fails_does_not_stop_a_calculation(self) -> None:
        # Whatever goes wrong in an estimate of the sphere is printed in its
        # place: the set-up goes on to the SCF, and a finished SCF to its
        # results.  The input stops after one step, with exit code 3.
        text = (DATA / "H_cli_smoke.in").read_text(encoding="utf-8")
        with patch.object(
            parsec_output, "domain_setup_lines", side_effect=KeyError("no table")
        ):
            code, _console, log = self.run_smoke(text, "--quiet")
        self.assertEqual(code, 3)
        lines = log.split("\n")
        self.assertIn(
            " --- Energy the sphere adds: no estimate, the report failed "
            "(KeyError: 'no table')",
            lines,
        )
        self.assertIn(" Density at the sphere: no estimate, the SCF did not converge", lines)
        with patch.object(
            parsec_output, "domain_finish_lines", side_effect=MemoryError("shells")
        ):
            code, _console, log = self.run_smoke(text, "--quiet")
        self.assertEqual(code, 3)
        lines = log.split("\n")
        self.assertIn(
            " Density at the sphere: no estimate, the report failed "
            "(MemoryError: shells)",
            lines,
        )
        self.assertFalse(any("Calculation failed" in line for line in lines))
        self.assertTrue(any("Total Energy =" in line for line in lines))

    def test_dry_run_says_what_the_rule_gives(self) -> None:
        text = (DATA / "H_cli_smoke.in").read_text(encoding="utf-8")
        code, console, log = self.run_smoke(text, "--dry-run")
        self.assertEqual((code, log), (0, ""))
        self.assertIn(
            " --- Without Boundary_Sphere_Radius the default rule 1 would give "
            "(set by no atomic density for H):\n"
            "       Boundary_Sphere_Radius: 7.5 ang\n"
            "       Hartree_Boundary_Tolerance: 1.0e-03 Ry\n",
            console,
        )
        code, console, log = self.run_smoke(parsed_inputs.without_radius(text), "--dry-run")
        self.assertEqual((code, log), (0, ""))
        self.assertIn(
            " --- Radius 7.5 ang chosen by the default rule 1 "
            "(Domain_Energy_Tolerance = 1.0E-03 Ry)\n",
            console,
        )
        self.assertIn("       Boundary_Sphere_Radius: 7.5 ang\n", console)


if __name__ == "__main__":
    unittest.main()
