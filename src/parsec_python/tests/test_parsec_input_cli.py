from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import zipfile
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import patch

import numpy as np

from parsec_python import (
    ANGSTROM_TO_BOHR,
    Atom,
    EnergyBreakdown,
    GridSettings,
    HartreeSettings,
    ParsecInputError,
    PeriodicGridSettings,
    PreparationTimings,
    RunTimings,
    SCFIteration,
    build_cluster_grid,
    parse_parsec_input,
    read_parsec_pseudopotential,
    summarize_translation,
)
import parsec_python.cli as cli_module
from parsec_python.cli import main as cli_main, save_result_archive
from parsec_python.Hartree import domain as domain_module


DATA = Path(__file__).parent / "data"
H2_INPUT = DATA / "H2_parsec.in"
SMOKE_INPUT = DATA / "H_cli_smoke.in"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
PHYSICAL_H2 = REPOSITORY_ROOT / "examples" / "h2_full_nonlocal"
PERIODIC_SILICON = REPOSITORY_ROOT / "examples" / "3d_Si" / "parsec.in"
# One hydrogen atom, without a line for a sphere radius.  ``{domain}`` takes
# the lines that make it a periodic cell or a cluster.
HYDROGEN_INPUT = """\
{domain}
Grid_Spacing: 0.6 bohr
Expansion_Order: 4
Coordinate_Unit: Cartesian_Bohr
States_Num: 3
Max_Iter: 40
Convergence_Criterion: 1e-5 Ry
Eigensolver: chebff
Mixing_Method: Anderson
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  3.0  3.0  3.0
end Atom_Coord
Correlation_Type: ca
"""
PERIODIC_DOMAIN = """\
Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  6.0  6.0  6.0
end Cell_Shape"""
CLUSTER_DOMAIN = """\
Boundary_Conditions: cluster
Cluster_Domain_Shape: sphere"""


class ParsecInputTests(unittest.TestCase):
    def parse_modified_h2(self, replacements: dict[str, str]) -> object:
        text = H2_INPUT.read_text(encoding="utf-8")
        for old, new in replacements.items():
            self.assertIn(old, text)
            text = text.replace(old, new)
        with patch.object(Path, "read_text", return_value=text):
            return parse_parsec_input(H2_INPUT)

    def test_exact_h2_input_translation(self) -> None:
        translation = parse_parsec_input(H2_INPUT)
        problem = translation.problem

        self.assertEqual(translation.source, H2_INPUT.resolve())
        self.assertEqual(len(problem.atoms), 2)
        self.assertEqual(tuple(atom.symbol for atom in problem.atoms), ("H", "H"))
        np.testing.assert_allclose(
            [atom.position for atom in problem.atoms],
            np.asarray(
                [
                    [0.0, 0.0, -0.375 * ANGSTROM_TO_BOHR],
                    [0.0, 0.0, 0.375 * ANGSTROM_TO_BOHR],
                ]
            ),
        )
        self.assertAlmostEqual(problem.grid.spacing, 0.2 * ANGSTROM_TO_BOHR)
        self.assertAlmostEqual(problem.grid.radius, 7.0 * ANGSTROM_TO_BOHR)
        self.assertEqual(problem.grid.expansion_order, 8)
        self.assertEqual(problem.grid.shift, (0.5, 0.5, 0.5))
        self.assertTrue(problem.recenter_geometry)

        hydrogen = problem.pseudopotentials["H"]
        self.assertEqual(hydrogen.path, (DATA / "H_POTRE.DAT").resolve())
        self.assertEqual(hydrogen.local_angular_momentum, 0)
        self.assertTrue(hydrogen.read_valence_density)

        self.assertEqual(problem.scf.number_of_states, 16)
        self.assertEqual(problem.scf.max_iterations, 50)
        self.assertAlmostEqual(problem.scf.convergence_criterion, 2.0e-4)
        self.assertEqual(problem.scf.fermi_temperature_kelvin, 500.0)
        self.assertEqual(problem.eigensolver.method, "chebff")
        self.assertEqual(problem.eigensolver.first_filter_degree, 10)
        self.assertEqual(problem.eigensolver.first_filter_cycles, 2)
        self.assertEqual(problem.eigensolver.matvec_block_size, 6)
        self.assertEqual(problem.eigensolver.subspace_buffer, 6)
        self.assertEqual(problem.eigensolver.filter_degree, 10)
        self.assertEqual(problem.eigensolver.filter_degree_delta, 0)
        self.assertAlmostEqual(problem.eigensolver.tolerance, 1.0e-4)
        self.assertAlmostEqual(problem.mixing.parameter, 0.15)
        self.assertEqual(problem.mixing.memory, 4)
        self.assertEqual(problem.mixing.restart, 20)
        self.assertEqual(problem.hartree.multipole_order, 9)
        self.assertEqual(problem.hartree.boundary_method, "auto")
        self.assertEqual(problem.scf.net_charge, 0.0)
        self.assertFalse(problem.scf.use_plain_residual)
        self.assertFalse(hydrogen.use_spline)
        self.assertTrue(translation.output_all_states)
        self.assertEqual(translation.output_level, 4)
        self.assertFalse(
            any("Chebdav_Degree" in item for item in translation.warnings)
        )

    def test_h2_parser_reproduces_parsec_grid_size(self) -> None:
        translation = parse_parsec_input(H2_INPUT)
        self.assertEqual(build_cluster_grid(translation.problem.grid).size, 179944)

    def test_parsec_filter_defaults_are_materialized(self) -> None:
        translation = self.parse_modified_h2(
            {
                "Chebdav_Degree: 10\n": "",
                "Chebyshev_Degree: 10\n": "",
                "Chebyshev_Degree_Delta: 0\n": "",
            }
        )
        settings = translation.problem.eigensolver

        self.assertEqual(settings.method, "chebff")
        self.assertEqual(settings.first_filter_degree, 20)
        self.assertEqual(settings.first_filter_cycles, 2)
        self.assertEqual(settings.matvec_block_size, 6)
        self.assertEqual(settings.filter_degree, 15)
        self.assertEqual(settings.filter_degree_delta, 3)
        self.assertEqual(settings.subspace_buffer, 6)
        self.assertEqual(translation.warnings, ())

    def test_invalid_parsec_filter_controls_are_reset_with_warnings(self) -> None:
        translation = self.parse_modified_h2(
            {
                "Chebdav_Degree: 10": (
                    "Chebdav_Degree: 4\n"
                    "FF_MaxIter: 12\n"
                    "Matvec_Blocksize: 3\n"
                    "Subspace_Buffer_Size: 2"
                ),
            }
        )
        settings = translation.problem.eigensolver

        self.assertEqual(settings.first_filter_degree, 15)
        self.assertEqual(settings.first_filter_cycles, 2)
        self.assertEqual(settings.matvec_block_size, 3)
        self.assertEqual(settings.subspace_buffer, 6)
        self.assertTrue(
            any("Chebdav_Degree=4" in item for item in translation.warnings)
        )
        self.assertTrue(
            any("FF_MaxIter=12" in item for item in translation.warnings)
        )
        self.assertTrue(
            any("Subspace_Buffer_Size=2" in item for item in translation.warnings)
        )

    def test_chebdav_and_arpack_are_not_collapsed_to_generic_methods(self) -> None:
        chebdav = self.parse_modified_h2(
            {
                "Eigensolver: chebff": "Eigensolver: chebdav",
                "Chebdav_Degree: 10": "Chebdav_Degree: 20",
            }
        )
        arpack = self.parse_modified_h2(
            {
                "Eigensolver: chebff": (
                    "Eigensolver: arpack\n"
                    "Subspace_Buffer_Size: 0"
                )
            }
        )

        self.assertEqual(chebdav.problem.eigensolver.method, "chebdav")
        self.assertEqual(chebdav.problem.eigensolver.matvec_block_size, 6)
        self.assertEqual(arpack.problem.eigensolver.method, "arpack")
        self.assertEqual(arpack.problem.eigensolver.matvec_block_size, 4)
        self.assertEqual(arpack.problem.eigensolver.subspace_buffer, 0)

    def test_chebdav_rejects_degree_below_parsec_minimum(self) -> None:
        with self.assertRaisesRegex(
            ParsecInputError,
            "Chebdav_Degree must be at least 15",
        ):
            self.parse_modified_h2(
                {"Eigensolver: chebff": "Eigensolver: chebdav"}
            )

    def test_matvec_blocksize_must_be_positive(self) -> None:
        with self.assertRaisesRegex(
            ParsecInputError,
            "Matvec_Blocksize must be positive",
        ):
            self.parse_modified_h2(
                {
                    "Chebdav_Degree: 10": (
                        "Chebdav_Degree: 10\nMatvec_Blocksize: 0"
                    ),
                }
            )

    def test_fixed_single_point_accepts_standard_relaxation_metadata(self) -> None:
        translation = self.parse_modified_h2(
            {
                "Atom_Types_Num: 1": (
                    "Old_Pseudopotential_Format: .false.\n"
                    "Periodic_System: .false.\n"
                    "Skip_force: .false.\n"
                    "Total_Atom_Num: 2\n"
                    "Ion_Energy_Diff: 0.644\n"
                    "Movement_Num: 100\n"
                    "Force_Min: 0.001\n"
                    "Max_Step: -1\n"
                    "Min_Step: 0.001\n\n"
                    "Atom_Types_Num: 1"
                )
            }
        )

        self.assertEqual(len(translation.problem.atoms), 2)
        self.assertTrue(translation.problem.recenter_geometry)
        self.assertTrue(
            any("fixed single point" in item for item in translation.warnings)
        )
        self.assertTrue(
            any("forces" in item for item in translation.warnings)
        )

    def test_periodic_flag_and_declared_atom_count_remain_strict(self) -> None:
        with self.assertRaisesRegex(ParsecInputError, "Periodic_System=true"):
            self.parse_modified_h2(
                {
                    "Boundary_Conditions: cluster": (
                        "Periodic_System: .true.\n"
                        "Boundary_Conditions: cluster"
                    )
                }
            )
        with self.assertRaisesRegex(ParsecInputError, "Total_Atom_Num=3"):
            self.parse_modified_h2(
                {
                    "Atom_Types_Num: 1": (
                        "Total_Atom_Num: 3\nAtom_Types_Num: 1"
                    )
                }
            )

    def test_translation_summary_has_resolved_potential(self) -> None:
        summary = summarize_translation(parse_parsec_input(H2_INPUT))
        self.assertIn("Atoms: 2", summary)
        self.assertIn("Species: H", summary)
        self.assertIn(str((DATA / "H_POTRE.DAT").resolve()), summary)
        self.assertIn("first_filter=10x2", summary)
        self.assertIn("block=6", summary)

    def test_full_physical_h2_potential_is_not_the_synthetic_fixture(self) -> None:
        translation = parse_parsec_input(PHYSICAL_H2 / "parsec.in")
        hydrogen_path = translation.problem.pseudopotentials["H"].path
        potential = read_parsec_pseudopotential(hydrogen_path)

        self.assertEqual(hydrogen_path, (PHYSICAL_H2 / "H_POTRE.DAT").resolve())
        self.assertGreater(hydrogen_path.stat().st_size, 100_000)
        self.assertEqual(potential.radii.size, 861)
        self.assertEqual(sorted(potential.channel_potentials), [0, 1])
        projector, sign = potential.radial_projector(1, 0)
        self.assertTrue(np.all(np.isfinite(projector)))
        self.assertEqual(sign, -1.0)


class PeriodicAndClusterInputTests(unittest.TestCase):
    """A periodic cell beside a cluster whose sphere radius may be left out."""

    def parse_text(self, text: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parsec.in"
            path.write_text(text, encoding="utf-8")
            return parse_parsec_input(path, pseudopotential_directory=DATA)

    def parse(self, domain: str, *lines: str):
        return self.parse_text(
            HYDROGEN_INPUT.format(domain=domain)
            + "".join(line + "\n" for line in lines)
        )

    def parse_periodic_without_the_rule(self, *lines: str):
        """Parse the periodic input; the default rule must not be asked."""

        with (
            patch.object(
                domain_module,
                "default_domain",
                side_effect=AssertionError("the default rule chose a radius"),
            ),
            patch.object(
                domain_module,
                "auto_boundary_tolerance",
                side_effect=AssertionError("the default rule chose a tolerance"),
            ),
        ):
            return self.parse(PERIODIC_DOMAIN, *lines)

    def test_periodic_input_has_no_sphere_for_the_default_rule(self) -> None:
        # No Boundary_Sphere_Radius line: for a cluster that asks the rule.
        translation = self.parse_periodic_without_the_rule()
        problem = translation.problem

        self.assertIsInstance(problem.grid, PeriodicGridSettings)
        np.testing.assert_array_equal(
            problem.periodic_cell.lattice_vectors, 6.0 * np.eye(3)
        )
        self.assertEqual(problem.grid.box_lengths, (6.0, 6.0, 6.0))
        self.assertFalse(problem.recenter_geometry)
        np.testing.assert_array_equal(problem.atoms[0].position, (3.0, 3.0, 3.0))
        self.assertIsNone(translation.domain)
        self.assertEqual(translation.domain_rule_seconds, 0.0)
        self.assertEqual(translation.boundary_tolerance_from, "default")
        self.assertEqual(translation.warnings, ())
        # The settings a periodic problem built without the parser has.
        self.assertEqual(problem.hartree, HartreeSettings())
        summary = summarize_translation(translation)
        self.assertIn("Grid: periodic, cell=", summary)
        self.assertNotIn("Hartree boundary", summary)

    def test_shipped_periodic_example_parses(self) -> None:
        translation = parse_parsec_input(PERIODIC_SILICON)
        problem = translation.problem

        self.assertEqual(len(problem.atoms), 8)
        np.testing.assert_array_equal(
            problem.periodic_cell.lattice_vectors, 10.26 * np.eye(3)
        )
        self.assertIsInstance(problem.grid, PeriodicGridSettings)
        self.assertIsNone(translation.domain)
        self.assertEqual(translation.boundary_tolerance_from, "default")

    def test_keywords_of_the_sphere_do_not_act_on_a_periodic_input(self) -> None:
        plain = self.parse_periodic_without_the_rule()
        # Each of these asks the default rule, or is refused beside a radius
        # left to it, in a cluster input.
        for lines in (
            ("Boundary_Sphere_Radius: auto",),
            ("Hartree_Boundary_Tolerance: auto",),
            ("Hartree_Boundary_Tolerance: off",),
            ("Hartree_Atomic_Tail: off",),
            ("Domain_Energy_Tolerance: 1e-5 Ry",),
            ("Cluster_Domain_Shape: sphere", "Solver_Lpole: 12"),
        ):
            with self.subTest(lines=lines):
                translation = self.parse_periodic_without_the_rule(*lines)
                self.assertEqual(translation.problem.grid, plain.problem.grid)
                np.testing.assert_array_equal(
                    translation.problem.periodic_cell.lattice_vectors,
                    plain.problem.periodic_cell.lattice_vectors,
                )
                self.assertIsNone(translation.domain)
                self.assertEqual(translation.domain_rule_seconds, 0.0)
                self.assertNotEqual(translation.boundary_tolerance_from, "rule")
                self.assertNotIn(
                    "Hartree boundary", summarize_translation(translation)
                )
        # They are read and checked like the other keywords of the input.
        for line in ("Hartree_Boundary_Tolerance: -1 Ry", "Hartree_Atomic_Tail: sometimes"):
            with self.subTest(line=line), self.assertRaises(ParsecInputError):
                self.parse(PERIODIC_DOMAIN, line)

    def test_cluster_input_without_a_radius_still_gets_the_default_rule(self) -> None:
        translation = self.parse(CLUSTER_DOMAIN)
        problem = translation.problem

        self.assertIsNone(problem.periodic_cell)
        self.assertIsInstance(problem.grid, GridSettings)
        choice = translation.domain
        self.assertIsNotNone(choice)
        self.assertEqual(problem.grid.radius, choice.radius)
        self.assertEqual(problem.hartree.boundary_tolerance, choice.boundary_tolerance)
        self.assertEqual(translation.boundary_tolerance_from, "rule")
        self.assertGreater(translation.domain_rule_seconds, 0.0)
        self.assertTrue(problem.recenter_geometry)
        self.assertIn("Hartree boundary: Solver_Lpole=9", summarize_translation(translation))
        # A radius left to the rule still excludes PARSEC's boundary.
        with self.assertRaisesRegex(ParsecInputError, "give Boundary_Sphere_Radius"):
            self.parse(CLUSTER_DOMAIN, "Hartree_Boundary_Tolerance: off")
        # With its radius the same input is parsed without the rule.
        given = self.parse(CLUSTER_DOMAIN, "Boundary_Sphere_Radius: 4.0 bohr")
        self.assertIsNone(given.domain)
        self.assertEqual(given.problem.grid.radius, 4.0)
        self.assertEqual(given.boundary_tolerance_from, "default")

    def test_what_upstream_refuses_of_a_periodic_input_stays_refused(self) -> None:
        periodic = HYDROGEN_INPUT.format(domain=PERIODIC_DOMAIN)
        for old, new, message in (
            (
                "Correlation_Type: ca",
                "Correlation_Type: pbe",
                "only supports Correlation_Type=CA",
            ),
            (
                "Periodic_System: .true.\n",
                "",
                "Periodic_System and Boundary_Conditions disagree",
            ),
            (
                "Boundary_Conditions: bulk",
                "Boundary_Conditions: slab",
                "wire and slab are not",
            ),
            (
                "begin Cell_Shape\n  6.0  6.0  6.0\nend Cell_Shape\n",
                "",
                "requires exactly one Cell_Shape block",
            ),
            (
                "  6.0  6.0  6.0\n",
                "  6.0  6.0\n",
                "three orthorhombic side lengths",
            ),
        ):
            self.assertIn(old, periodic)
            with self.subTest(message=message):
                with self.assertRaisesRegex(ParsecInputError, message):
                    self.parse_text(periodic.replace(old, new))


class CommandLineTests(unittest.TestCase):
    def test_package_cli_dry_run(self) -> None:
        self.assertEqual(cli_main([str(H2_INPUT), "--dry-run"]), 0)

    def test_package_folder_main_dry_run(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(PACKAGE_ROOT / "main.py"),
                str(H2_INPUT),
                "--dry-run",
            ],
            cwd=PACKAGE_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Dry run successful", completed.stdout)
        self.assertNotIn("Traceback", completed.stderr)

    def test_missing_input_is_a_clean_input_error(self) -> None:
        error = StringIO()
        with redirect_stderr(error):
            return_code = cli_main([str(DATA / "does_not_exist.in"), "--dry-run"])
        self.assertEqual(return_code, 2)
        self.assertIn("Input error: cannot read PARSEC input", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())

    def test_log_archive_collision_is_rejected_before_calculation(self) -> None:
        error = StringIO()
        with redirect_stderr(error):
            return_code = cli_main(
                [
                    str(SMOKE_INPUT),
                    "--log",
                    "same-output.npz",
                    "--output",
                    "same-output.npz",
                ]
            )
        self.assertEqual(return_code, 2)
        self.assertIn("--log and --output resolve to the same path", error.getvalue())

    def test_actual_one_iteration_cli_and_result_archive(self) -> None:
        messages: list[str] = []
        log_paths: list[Path] = []

        class MemoryLog:
            def __init__(self, path: Path, quiet: bool = False) -> None:
                self.quiet = quiet
                log_paths.append(path)

            def __enter__(self) -> "MemoryLog":
                return self

            def __exit__(self, _type, _value, _traceback) -> None:
                return None

            def write(self, message: str = "") -> None:
                messages.append(message)

        reported_archive = DATA / "calculation.npz"
        with (
            patch.object(cli_module, "_RunLog", MemoryLog),
            patch.object(
                cli_module,
                "save_result_archive",
                return_value=reported_archive,
            ) as archive_writer,
            patch.dict("os.environ", {"PARSEC_HARTREE_LPOLE": "9"}),
        ):
            return_code = cli_main(
                [
                    str(SMOKE_INPUT),
                    "--output",
                    "calculation",
                    "--quiet",
                ]
            )

        # One iteration is deliberately too short to converge.
        self.assertEqual(return_code, 3)
        self.assertEqual(log_paths, [SMOKE_INPUT.parent / "parsec.out"])
        report = "\n".join(messages)
        # The header echoes the input, and a boundary switch of the
        # accelerated driver is named as ignored.
        self.assertIn(" Hartree boundary tolerance in the input is : 1.000000E-03  Ry", report)
        self.assertIn(
            "WARNING: PARSEC_HARTREE_LPOLE=9 is a switch of the accelerated driver "
            "and is ignored by the reference driver", report)
        self.assertIn("PARSEC-PYTHON - Modular real-space DFT program", report)
        self.assertIn("Performing Chebyshev subspace filtering", report)
        self.assertNotIn("Performing Lanczos/ARPACK diagonalization", report)
        self.assertIn("Full active grid points", report)
        self.assertIn("State   Eigenvalue [Ry]", report)
        self.assertIn("Eigenvalue Energy", report)
        self.assertIn("SRE of pot. & charge weighted pot", report)
        self.assertIn("Setup timings [sec]", report)
        self.assertIn("Finite-difference construction", report)
        self.assertIn("SCF timing analysis [sec]", report)
        self.assertIn("Hamiltonian binding subtotal", report)
        self.assertIn("Exchange-correlation subtotal", report)
        self.assertIn("Maximum SCF iterations reached", report)
        archive_writer.assert_called_once()
        result = archive_writer.call_args.args[1]
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.atoms[0].symbol, "H")

    def test_periodic_run_names_no_switch_of_a_hartree_boundary(self) -> None:
        # A periodic cell has no Hartree boundary values, so there is no
        # boundary of this run that a switch of the accelerated driver could
        # be said to leave alone: the log is the one without the switches.
        switches = {
            "PARSEC_HARTREE_BOUNDARY": "legacy",
            "PARSEC_HARTREE_BOUNDARY_TOLERANCE": "1e-4",
            "PARSEC_HARTREE_ATOMIC_TAIL": "on",
            "PARSEC_HARTREE_LPOLE": "12",
        }
        text = HYDROGEN_INPUT.format(domain=PERIODIC_DOMAIN).replace(
            "Max_Iter: 40", "Max_Iter: 1"
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "parsec.in"
            source.write_text(text, encoding="utf-8")
            with patch.dict("os.environ", switches):
                return_code = cli_main(
                    [str(source), "--pp-dir", str(DATA), "--no-archive", "--quiet"]
                )
            report = (Path(directory) / "parsec.out").read_text(encoding="utf-8")

        # One iteration is deliberately too short to converge.
        self.assertEqual(return_code, 3)
        self.assertIn("Maximum SCF iterations reached", report)
        self.assertNotIn("is a switch of the accelerated driver", report)
        self.assertNotIn("WARNING", report)
        self.assertNotIn("Hartree boundary", report)

    def test_archive_suffix_and_reproducibility_fields(self) -> None:
        energy = EnergyBreakdown(
            eigenvalue=0.0,
            hartree=0.0,
            integral_vxc_rho=0.0,
            exchange_correlation=0.0,
            electron_ion=0.0,
            ion_ion=0.0,
            electronic=0.0,
            total=0.0,
        )
        result = SimpleNamespace(
            atoms=(Atom("H", [0.0, 0.0, 0.0]),),
            grid=SimpleNamespace(
                coordinates=np.zeros((2, 3)),
                integer_coordinates=np.zeros((2, 3), dtype=np.int64),
            ),
            energies=energy,
            history=[
                SCFIteration(
                    iteration=1,
                    weighted_residual=1.0,
                    plain_residual=1.0,
                    eigen_residual_max=1.0,
                    hartree_residual=1.0,
                    energies=energy,
                )
            ],
            density=np.zeros(2),
            core_density=np.zeros(2),
            ionic_potential=np.zeros(2),
            hartree_potential=np.zeros(2),
            xc_potential=np.zeros(2),
            input_effective_potential=np.zeros(2),
            output_effective_potential=np.zeros(2),
            next_effective_potential=np.zeros(2),
            eigenvalues=np.zeros(1),
            occupations=np.zeros(1),
            fermi_level=0.0,
            electron_count=1.0,
            converged=False,
            iterations=1,
            wavefunctions=np.zeros((2, 1)),
            atomic_reference_correction=-2.0,
            all_electron_total=-2.0,
            timings=RunTimings(
                preparation=PreparationTimings(
                    finite_difference_seconds=0.25,
                    total_seconds=0.5,
                ),
                hamiltonian_binding_seconds=0.01,
                total_seconds=1.5,
            ),
        )
        with patch.object(cli_module.np, "savez") as writer:
            saved = save_result_archive(PACKAGE_ROOT / "unsuffixed_result", result)

        self.assertEqual(saved.suffix, ".npz")
        self.assertEqual(writer.call_args.args[0], saved)
        payload = writer.call_args.kwargs
        self.assertEqual(payload["atom_symbols"].tolist(), ["H"])
        self.assertAlmostEqual(
            float(payload["atomic_reference_correction_ry"]), -2.0
        )
        self.assertAlmostEqual(float(payload["all_electron_total_ry"]), -2.0)
        np.testing.assert_allclose(
            payload["atom_coordinates_bohr"], [[0.0, 0.0, 0.0]]
        )
        self.assertEqual(payload["scf_timing_history"].shape, (1, 8))
        np.testing.assert_array_equal(payload["representations"], [1])
        self.assertEqual(
            payload["scf_timing_history_columns"].tolist(),
            [
                "iteration",
                "hamiltonian_binding_seconds",
                "diagonalization_seconds",
                "occupations_density_seconds",
                "hartree_seconds",
                "xc_seconds",
                "mixing_energy_seconds",
                "total_seconds",
            ],
        )
        self.assertAlmostEqual(
            float(payload["timing_scf_hamiltonian_binding_seconds"]), 0.01
        )
        self.assertAlmostEqual(
            float(payload["timing_preparation_finite_difference_seconds"]),
            0.25,
        )

        # Result-like objects created before timing metadata was introduced
        # remain archive-compatible.
        del result.timings
        with patch.object(cli_module.np, "savez") as legacy_writer:
            save_result_archive(PACKAGE_ROOT / "legacy_unsuffixed_result", result)
        legacy_payload = legacy_writer.call_args.kwargs
        self.assertIn("scf_timing_history", legacy_payload)
        self.assertNotIn("timing_scf_total_seconds", legacy_payload)

    def test_archive_is_stored_uncompressed_with_the_compressed_content(self) -> None:
        generator = np.random.default_rng(5)
        points = 301
        energy = EnergyBreakdown(
            eigenvalue=-1.25,
            hartree=0.5,
            integral_vxc_rho=-0.75,
            exchange_correlation=-0.6,
            electron_ion=-2.0,
            ion_ion=0.3,
            electronic=-3.1,
            total=-2.8,
        )
        result = SimpleNamespace(
            atoms=(Atom("H", [0.0, 0.0, 0.4]), Atom("H", [0.0, 0.0, -0.4])),
            grid=SimpleNamespace(
                coordinates=generator.standard_normal((points, 3)),
                integer_coordinates=generator.integers(
                    -40, 40, size=(points, 3), dtype=np.int64
                ),
            ),
            energies=energy,
            history=[
                SCFIteration(
                    iteration=1,
                    weighted_residual=1.0e-2,
                    plain_residual=2.0e-2,
                    # CHEBFF reports no eigen-residual on the first step.
                    eigen_residual_max=float("nan"),
                    hartree_residual=1.0e-9,
                    energies=energy,
                )
            ],
            density=generator.random(points),
            core_density=generator.random(points),
            ionic_potential=generator.standard_normal(points),
            hartree_potential=generator.standard_normal(points),
            xc_potential=generator.standard_normal(points),
            input_effective_potential=generator.standard_normal(points),
            output_effective_potential=generator.standard_normal(points),
            next_effective_potential=generator.standard_normal(points),
            eigenvalues=np.sort(generator.standard_normal(4)),
            occupations=np.asarray((1.0, 1.0, 0.5, 0.0)),
            representations=np.asarray((1, 3, 2, 1), dtype=np.int32),
            fermi_level=-0.2,
            electron_count=5.0,
            converged=True,
            iterations=1,
            wavefunctions=generator.standard_normal((points, 4)),
        )
        full_grid_members = (
            "coordinates_bohr",
            "integer_coordinates",
            "density_e_per_bohr3",
            "core_density_e_per_bohr3",
            "ionic_potential_ry",
            "hartree_potential_ry",
            "xc_potential_ry",
            "input_effective_potential_ry",
            "output_effective_potential_ry",
            "next_effective_potential_ry",
        )
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            stored_path = save_result_archive(
                Path(directory) / "stored", result, include_wavefunctions=True
            )
            # The same members through the former compressing writer.
            with patch.object(cli_module.np, "savez", np.savez_compressed):
                compressed_path = save_result_archive(
                    Path(directory) / "compressed",
                    result,
                    include_wavefunctions=True,
                )
            with zipfile.ZipFile(stored_path) as archive:
                stored_members = archive.infolist()
            with zipfile.ZipFile(compressed_path) as archive:
                compressed_members = archive.infolist()
            self.assertEqual(
                [item.filename for item in stored_members],
                [item.filename for item in compressed_members],
            )
            for item in stored_members:
                self.assertEqual(item.compress_type, zipfile.ZIP_STORED)
            for item in compressed_members:
                self.assertEqual(item.compress_type, zipfile.ZIP_DEFLATED)

            with (
                np.load(stored_path, allow_pickle=False) as stored,
                np.load(compressed_path, allow_pickle=False) as compressed,
            ):
                self.assertEqual(stored.files, compressed.files)
                for name in full_grid_members + ("wavefunctions",):
                    self.assertIn(name, stored.files)
                for name in compressed.files:
                    with self.subTest(member=name):
                        self.assertEqual(stored[name].dtype, compressed[name].dtype)
                        self.assertEqual(stored[name].shape, compressed[name].shape)
                        self.assertEqual(
                            stored[name].tobytes(), compressed[name].tobytes()
                        )
                # The values are the result's own arrays, bit for bit.
                np.testing.assert_array_equal(
                    stored["coordinates_bohr"], result.grid.coordinates
                )
                np.testing.assert_array_equal(
                    stored["integer_coordinates"], result.grid.integer_coordinates
                )
                np.testing.assert_array_equal(
                    stored["density_e_per_bohr3"], result.density
                )
                np.testing.assert_array_equal(
                    stored["wavefunctions"], result.wavefunctions
                )


if __name__ == "__main__":
    unittest.main()
