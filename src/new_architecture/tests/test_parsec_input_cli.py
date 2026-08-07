from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import patch

import numpy as np

from new_architecture import (
    ANGSTROM_TO_BOHR,
    Atom,
    EnergyBreakdown,
    ParsecInputError,
    SCFIteration,
    build_cluster_grid,
    parse_parsec_input,
    read_parsec_pseudopotential,
    summarize_translation,
)
import new_architecture.cli as cli_module
from new_architecture.cli import main as cli_main, save_result_archive


DATA = Path(__file__).parent / "data"
H2_INPUT = DATA / "H2_parsec.in"
SMOKE_INPUT = DATA / "H_cli_smoke.in"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_H2 = PACKAGE_ROOT / "benchmarks" / "h2_full_nonlocal"


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
        self.assertIn("PARSEC-PYTHON - Modular real-space DFT program", report)
        self.assertIn("Performing Chebyshev subspace filtering", report)
        self.assertNotIn("Performing Lanczos/ARPACK diagonalization", report)
        self.assertIn("Full active grid points", report)
        self.assertIn("State   Eigenvalue [Ry]", report)
        self.assertIn("Eigenvalue Energy", report)
        self.assertIn("SRE of pot. & charge weighted pot", report)
        self.assertIn("Maximum SCF iterations reached", report)
        archive_writer.assert_called_once()
        result = archive_writer.call_args.args[1]
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.atoms[0].symbol, "H")

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
        )
        with patch.object(cli_module.np, "savez_compressed") as writer:
            saved = save_result_archive(PACKAGE_ROOT / "unsuffixed_result", result)

        self.assertEqual(saved.suffix, ".npz")
        self.assertEqual(writer.call_args.args[0], saved)
        payload = writer.call_args.kwargs
        self.assertEqual(payload["atom_symbols"].tolist(), ["H"])
        np.testing.assert_allclose(
            payload["atom_coordinates_bohr"], [[0.0, 0.0, 0.0]]
        )


if __name__ == "__main__":
    unittest.main()
