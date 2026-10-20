"""Initial structures from machine-learned force fields (``parsec_python.MLFF``).

The force field of these tests is ``tests/data/mlff_springs.py``, an ASE
calculator of springs, run through ``MLFF_Model: python:file.py:function``
by the interpreter of the tests: the worker process, the files, the reuse of
a relaxed structure, the point group, fixed atoms and the drivers are those
of a real model.  UMA and MACE themselves are checked by
``mlff.py check`` where they are installed.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python.Forces import atom_symmetry_operations
from parsec_python.Input import ParsecInputError, parse_parsec_input
from parsec_python.MLFF import (
    STRUCTURE_FILE,
    TRAJECTORY_FILE,
    MLFFError,
    MLFFSettings,
    apply_mlff_option,
    read_structure_file,
    relax_initial_structure,
)
from parsec_python.MLFF import worker
from parsec_python.MLFF.cli import main as mlff_main
from parsec_python.MLFF.run import BOHR_TO_ANGSTROM
from parsec_python.models import Atom
from parsec_python.tests.unit.test_relaxation import (
    _Reporter,
    _run_cli,
    _ScratchCase,
    _SurfaceBackend,
    _water_lines,
    cli_module,
)

HAS_ASE = importlib.util.find_spec("ase") is not None
SPRINGS = Path(__file__).resolve().parents[1] / "data" / "mlff_springs.py"
TOY = f"python:{SPRINGS}:make_calculator"


def _positions(translation) -> np.ndarray:
    return np.array([atom.position for atom in translation.problem.atoms], dtype=np.float64)


class KeywordTests(_ScratchCase):
    def test_defaults(self) -> None:
        translation = self.parse(_water_lines("Initial_Structure: mlff"))
        self.assertEqual(translation.mlff, MLFFSettings())
        self.assertEqual(translation.mlff.model, "uma-s-1p2")
        self.assertIsNone(self.parse(_water_lines()).mlff)

    def test_every_keyword(self) -> None:
        translation = self.parse(
            _water_lines(
                "Initial_Structure: MLFF\nMLFF_Model: MACE-OFF-Medium\nMLFF_Task: omol\n"
                "MLFF_Fmax: 0.01 Ry/bohr\nMLFF_Steps: 40\nMLFF_Optimizer: FIRE\n"
                "MLFF_Device: cpu\nMLFF_Charge: -1\nMLFF_Spin: 2\nMLFF_Reuse: .false."
            )
        )
        settings = translation.mlff
        self.assertEqual(settings.model, "mace-off-medium")
        self.assertAlmostEqual(settings.fmax, 0.01 * 13.605693122994 / 0.529177, places=9)
        self.assertEqual(
            (settings.steps, settings.optimizer, settings.device, settings.charge, settings.spin),
            (40, "fire", "cpu", -1, 2),
        )
        self.assertFalse(settings.reuse)
        self.assertEqual(self.parse(_water_lines("Initial_Structure: mlff\nMLFF_Fmax: 0.02")).mlff.fmax, 0.02)

    def test_paths_are_taken_beside_the_input(self) -> None:
        translation = self.parse(
            _water_lines(
                "Initial_Structure: mlff\nMLFF_Model: models/my.model\n"
                "MLFF_Python: env/bin/python"
            )
        )
        self.assertEqual(Path(translation.mlff.model), self.directory / "models" / "my.model")
        self.assertEqual(translation.mlff.python, self.directory / "env" / "bin" / "python")
        translation = self.parse(_water_lines("Initial_Structure: mlff\nMLFF_Model: python:toy.py:make"))
        self.assertEqual(translation.mlff.model, f"python:{self.directory / 'toy.py'}:make")

    def test_mistakes_stop_in_the_parser(self) -> None:
        for control, message in (
            ("Initial_Structure: guess", "Initial_Structure must be input or mlff"),
            ("Initial_Structure: mlff\nMLFF_Optimizer: cg", "MLFF_Optimizer must be one of"),
            ("Initial_Structure: mlff\nMLFF_Device: tpu", "MLFF_Device must be one of"),
            ("Initial_Structure: mlff\nMLFF_Fmax: 0", "MLFF_Fmax must be positive"),
            ("Initial_Structure: mlff\nMLFF_Spin: 0", "MLFF_Spin is a multiplicity"),
            ("Initial_Structure: mlff\nMLFF_Python: not-a-python-xyz", "is not on the PATH"),
        ):
            with self.subTest(control=control):
                with self.assertRaisesRegex(ParsecInputError, message):
                    self.parse(_water_lines(control))

    def test_keywords_without_the_switch_are_named(self) -> None:
        translation = self.parse(_water_lines("MLFF_Model: mace-omol"))
        self.assertIsNone(translation.mlff)
        self.assertIn("without Initial_Structure: mlff", "\n".join(translation.warnings))

    def test_a_dry_run_does_not_start_the_force_field(self) -> None:
        # A model that cannot be built: a dry run must not try to.
        from parsec_python.acceleration.cli import main as accelerated_main

        source = self.write_input(
            _water_lines("Initial_Structure: mlff\nMLFF_Model: python:no_such_file.py:make")
        )
        for main in (None, accelerated_main):
            with self.subTest(driver="reference" if main is None else "accelerated"):
                arguments = [source, "--dry-run"]
                code, output, error = _run_cli(arguments) if main is None else _run_cli(arguments, main=main)
                self.assertEqual(code, 0, error)
                self.assertIn("Initial structure: relaxed with the force field", output)
                self.assertNotIn("MLFF error", output + error)

    def test_command_line_option(self) -> None:
        translation = self.parse(_water_lines())
        self.assertEqual(apply_mlff_option(translation, "").mlff, MLFFSettings())
        self.assertEqual(apply_mlff_option(translation, "mace-omol").mlff.model, "mace-omol")


class WorkerTests(unittest.TestCase):
    def test_model_families(self) -> None:
        for name, backend in (
            ("uma-s-1p1", "fairchem"),
            ("UMA-M-1P1", "fairchem"),
            ("esen-sm-conserving-all-omol", "fairchem"),
            ("/models/checkpoint.pt", "fairchem"),
            ("mace-omol", "mace"),
            ("mace-off-medium", "mace"),
            ("mace-mp-medium-mpa-0", "mace"),
            ("my.model", "mace"),
            ("python:pkg.module:make", "python"),
        ):
            with self.subTest(name=name):
                self.assertEqual(worker.backend_of(name), backend)
        with self.assertRaisesRegex(worker.ModelError, "unknown MLFF_Model"):
            worker.backend_of("chgnet")

    def test_point_group_averages_forces_and_positions(self) -> None:
        # Mirror x -> -x exchanging atoms 1 and 2.
        group = worker.PointGroup(
            [np.eye(3).tolist(), np.diag([-1.0, 1.0, 1.0]).tolist()], [[0, 1, 2], [0, 2, 1]]
        )
        forces = np.array([[0.3, 0.1, 0.0], [1.0, 0.0, 0.0], [-0.8, 0.2, 0.0]])
        group.adjust_forces(None, forces)
        np.testing.assert_allclose(forces, [[0.0, 0.1, 0.0], [0.9, 0.1, 0.0], [-0.9, 0.1, 0.0]])

    @unittest.skipUnless(HAS_ASE, "ASE is not installed")
    def test_python_factory_errors(self) -> None:
        with self.assertRaisesRegex(worker.ModelError, "does not exist"):
            worker.build_calculator("python:/no/such/file.py:make", device="cpu", task="omol", charge=0, spin=1)
        with self.assertRaisesRegex(worker.ModelError, "has no function"):
            worker.build_calculator(f"python:{SPRINGS}:absent", device="cpu", task="omol", charge=0, spin=1)


@unittest.skipUnless(HAS_ASE, "ASE is not installed")
class RelaxationTests(_ScratchCase):
    def translation(self, control="", **rows):
        lines = _water_lines(f"Initial_Structure: mlff\nMLFF_Model: {TOY}\n{control}", **rows)
        return parse_parsec_input(self.write_input(lines))

    def test_structure_files_and_reuse(self) -> None:
        translation = self.translation()
        relaxed, notes = relax_initial_structure(translation, self.directory)
        text = "\n".join(notes)
        self.assertIn("Initial structure relaxed with the force field", text)
        self.assertIn("4 symmetry operations kept", text)
        positions = _positions(relaxed)
        # The springs put O-H at the sum of the covalent radii, 0.97 angstrom.
        bond = np.linalg.norm(positions[1] - positions[0]) * BOHR_TO_ANGSTROM
        self.assertAlmostEqual(bond, 0.97, delta=0.02)
        key, symbols, stored, fields = read_structure_file(self.directory / STRUCTURE_FILE)
        self.assertEqual(symbols, ["O", "H", "H"])
        np.testing.assert_allclose(stored / BOHR_TO_ANGSTROM, positions, rtol=0, atol=1e-9)
        self.assertEqual(fields["converged"], "True")
        frames = (self.directory / TRAJECTORY_FILE).read_text().count("Properties=")
        self.assertGreater(frames, 1)
        # Same input and settings: the file is taken, the worker does not run.
        with patch("parsec_python.MLFF.run.run_worker", side_effect=AssertionError("ran")):
            again, notes = relax_initial_structure(translation, self.directory)
        self.assertIn(f"Initial structure from {STRUCTURE_FILE}", "\n".join(notes))
        np.testing.assert_array_equal(_positions(again), positions)
        # Other settings, or MLFF_Reuse: false, relax again.
        for control in ("MLFF_Fmax: 0.01", "MLFF_Reuse: .false."):
            with self.subTest(control=control):
                calls = []
                with patch("parsec_python.MLFF.run.run_worker", side_effect=lambda *a: calls.append(a) or {
                    "positions": (positions * BOHR_TO_ANGSTROM).tolist(), "energies": [0.0], "fmax": [0.0],
                    "steps": 0, "converged": True, "model": "toy", "device": "cpu", "charge": 0, "spin": 1,
                    "seconds": {"load": 0.0, "relax": 0.0}}):
                    relax_initial_structure(self.translation(control), self.directory)
                self.assertEqual(len(calls), 1)

    def test_the_point_group_of_the_run_is_kept(self) -> None:
        # A push along x on oxygen breaks the mirror x -> -x of the molecule.
        with patch.dict(os.environ, {"PARSEC_MLFF_TEST_PUSH": "0.5"}):
            kept, _ = relax_initial_structure(self.translation(), self.directory)
            lost, notes = relax_initial_structure(
                self.translation("Ignore_Symmetry: .true.\nMLFF_Reuse: .false."), self.directory
            )
        self.assertEqual(len(atom_symmetry_operations(kept.problem.atoms)), 4)
        self.assertIn("no symmetry imposed", "\n".join(notes))
        self.assertLess(len(atom_symmetry_operations(lost.problem.atoms)), 4)

    def test_fixed_atoms_stay(self) -> None:
        translation = self.translation(
            "Minimization: bfgs", oxygen_flag="0", hydrogen_flag="1",
            hydrogen=("1.6 0.6 0.0", "-1.6 0.4 0.0"),
        )
        start = _positions(translation)
        relaxed, notes = relax_initial_structure(translation, self.directory)
        self.assertIn("1 atoms fixed", "\n".join(notes))
        moved = _positions(relaxed)
        # Positions are compared after the centring the run applies.
        np.testing.assert_allclose(moved[2] - moved[0], start[2] - start[0], atol=0.2)
        self.assertGreater(np.abs(moved[1] - start[1] - (moved[0] - start[0])).max(), 1e-3)

    def test_restart_and_manual_runs_keep_their_structures(self) -> None:
        translation = self.translation()
        same, notes = relax_initial_structure(translation, self.directory, restarted=True)
        self.assertIs(same, translation)
        self.assertIn("continues from its restart structure", notes[0])
        manual = self.translation("Minimization: manual\nMovement_Num: 0")
        same, notes = relax_initial_structure(manual, self.directory)
        self.assertIs(same, manual)
        self.assertIn("manual run", notes[0])

    def test_a_model_that_does_not_describe_the_structure_is_refused(self) -> None:
        # Forces of thousands of eV/angstrom: the structure of the input stays.
        with patch.dict(os.environ, {"PARSEC_MLFF_TEST_PUSH": "5000"}):
            with self.assertRaisesRegex(MLFFError, "does not describe this structure"):
                relax_initial_structure(self.translation("Ignore_Symmetry: .true.\nMLFF_Steps: 3"), self.directory)
        self.assertFalse((self.directory / STRUCTURE_FILE).exists())
        self.assertTrue((self.directory / TRAJECTORY_FILE).exists())

    def test_errors_name_the_cure(self) -> None:
        translation = self.translation()
        with self.assertRaisesRegex(MLFFError, "cannot be started"):
            relax_initial_structure(
                replace(translation, mlff=replace(translation.mlff, python=self.directory / "no-python")),
                self.directory,
            )
        if importlib.util.find_spec("mace") is None:
            with self.assertRaisesRegex(MLFFError, "pip install mace-torch"):
                relax_initial_structure(
                    replace(translation, mlff=replace(translation.mlff, model="mace-off-medium")),
                    self.directory,
                )

    def test_drivers_start_from_the_relaxed_structure(self) -> None:
        surface = _SurfaceBackend()
        source = self.write_input(
            _water_lines(f"Initial_Structure: mlff\nMLFF_Model: {TOY}\nMinimization: BFGS\nMovement_Num: 0")
        )
        with (
            patch.object(cli_module, "reference_backend", surface.backend),
            patch.object(cli_module, "ParsecTextReporter", _Reporter),
        ):
            code, output, error = _run_cli([source, "--no-archive", "--quiet"])
        # Movement_Num 0: one structure, not relaxed on the springs (exit code 4).
        self.assertEqual((code, error), (4, ""))
        _key, _symbols, stored, _fields = read_structure_file(self.directory / STRUCTURE_FILE)
        np.testing.assert_allclose(surface.scf_calls[0][0], stored / BOHR_TO_ANGSTROM, rtol=0, atol=1e-9)
        log = (self.directory / "parsec.out").read_text(encoding="utf-8")
        self.assertIn("Initial structure relaxed with the force field", log)
        # The command-line option does the same for an input without the keyword.
        (self.directory / STRUCTURE_FILE).unlink()
        source = self.write_input(_water_lines("Minimization: BFGS\nMovement_Num: 0"))
        surface = _SurfaceBackend()
        with (
            patch.object(cli_module, "reference_backend", surface.backend),
            patch.object(cli_module, "ParsecTextReporter", _Reporter),
        ):
            code, output, error = _run_cli([source, "--no-archive", "--quiet", "--mlff", TOY])
        self.assertEqual((code, error), (4, ""))
        self.assertTrue((self.directory / STRUCTURE_FILE).is_file())

    def test_command_relax(self) -> None:
        source = self.write_input(_water_lines())
        code, output, error = _run_cli(["relax", source, "--model", TOY], main=mlff_main)
        self.assertEqual((code, error), (0, ""))
        self.assertIn("--restart-from", output)
        self.assertTrue((self.directory / STRUCTURE_FILE).is_file())
        code, output, error = _run_cli(["models", "--python", sys.executable], main=mlff_main)
        self.assertEqual(code, 0)
        self.assertIn("mace-omol", output)


if __name__ == "__main__":
    unittest.main()
