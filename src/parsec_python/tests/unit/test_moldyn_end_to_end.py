"""Molecular dynamics of the readable path against Fortran PARSEC, end to end.

Two short runs of a water molecule go through the command-line driver
(``parsec_python.cli.main``) with a self-consistent calculation behind every
structure, and the files they write are compared with those of Fortran
PARSEC for the same input (``tests/data/moldyn_parsec_end_to_end_reference.json``):

``langevin``
    a fresh Langevin run with PARSEC's own random generator, a temperature
    stair at every step and one fixed atom;
``restart``
    a microcanonical run continued from an ``mdinit.dat`` that Fortran PARSEC
    wrote.  A PARSEC built by gfortran cannot read that file itself, so the
    reference is the unbroken Fortran run, whose later steps the restart
    has to reproduce.

Both codes converge the SCF loop to 1e-8 Ry.  The numbers of a file agree
then to about one unit of its last printed digit; where a file prints more
digits than the forces have (``md_mech.dat``), the tolerance is that of
forces good to 1e-6 Ry/bohr.  The integrator alone is compared with the
Fortran routines bit for bit in ``test_moldyn.py``.

The sphere of 3.5 bohr is as small as the atoms allow, because the readable
path spends most of its time on the Hartree boundary values, whose number
grows with the surface of the sphere.
"""

from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest

import numpy as np

from parsec_python.Input import parse_parsec_input
from parsec_python.cli import main as cli_main

DATA = Path(__file__).resolve().parents[1] / "data"
REFERENCE = json.loads(
    (DATA / "moldyn_parsec_end_to_end_reference.json").read_text(encoding="utf-8")
)
CASES = REFERENCE["cases"]
FILES = ("md_nrg.dat", "md_mech.dat", "atom.cor", "movie.xyz")

#: ``md_nrg.dat`` prints energies with five decimals (eV per atom) and
#: temperatures with two (K).
ENERGY_TOLERANCE = 2.0e-5
TEMPERATURE_TOLERANCE = 2.0e-2
#: ``md_mech.dat`` prints eight significant digits.  Positions in bohr;
#: velocities and accelerations in atomic units: a force error of 1e-6
#: Ry/bohr on a hydrogen atom is an acceleration error of 3e-10.
POSITION_TOLERANCE = 5.0e-7
VELOCITY_TOLERANCE = 1.0e-8
ACCELERATION_TOLERANCE = 1.0e-9
#: ``atom.cor`` prints nine decimals (bohr), six in the blocks of a restart
#: and of the last step; ``movie.xyz`` six decimals (angstrom).
ATOM_COR_TOLERANCE = 1.0e-6
MOVIE_TOLERANCE = 2.0e-6

_STEP = re.compile(r"^\s*Step\s*#\s*(\d+)\s*$")
_COORDINATES = re.compile(r"^\s*Coordinates for step #\s*(\d+)\s*$")
_NUMBER = re.compile(r"\d\.\d")


def _write_case(case: dict, directory: Path, *, restart_file: bool = True) -> Path:
    """Write ``parsec.in``, the pseudopotentials and ``mdinit.dat`` of a case."""

    lines = [*case["input"], "Hartree_Boundary_Tolerance: off"]
    (directory / "parsec.in").write_text("\n".join(lines) + "\n")
    for symbol, name in REFERENCE["pseudopotentials"].items():
        shutil.copyfile(DATA / name, directory / f"{symbol}_POTRE.DAT")
    if restart_file and "mdinit" in case:
        (directory / "mdinit.dat").write_text("\n".join(case["mdinit"]) + "\n")
    return directory / "parsec.in"


def _run_case(case: dict, *arguments: str):
    """Run a case through the driver.

    Returns the exit status, the lines of every file of the run and the
    arrays of the result archive (empty with ``--no-archive``).
    """

    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        status = cli_main([str(_write_case(case, folder)), "--quiet", *arguments])
        files = {
            name: (folder / name).read_text().splitlines()
            for name in (*FILES, "parsec.out")
            if (folder / name).is_file()
        }
        archive = {}
        if (folder / "parsec_python_results.npz").is_file():
            with np.load(folder / "parsec_python_results.npz") as stored:
                archive = {name: stored[name] for name in stored.files}
    return status, files, archive


def _md_nrg(lines) -> dict:
    """``{step: [time, kinetic, potential, total, target, temperature]}``."""

    table = {}
    for line in lines[1:]:
        fields = line.split()
        table[int(fields[0])] = np.array([float(field) for field in fields[1:]])
    return table


def _md_mech(lines) -> dict:
    """``{step: array (atoms, 13)}``, read by column as ``13(1x,e15.8)``."""

    table, step = {}, None
    for line in lines:
        match = _STEP.match(line)
        if match:
            step = int(match.group(1))
            table[step] = []
        elif step is not None:
            table[step].append(
                [float(line[16 * k + 1 : 16 * k + 16]) for k in range(13)]
            )
    return {step: np.array(rows) for step, rows in table.items()}


def _atom_cor(lines) -> dict:
    """``{step: array (atoms, 3)}``; the closing block has the key ``"next"``."""

    table, key = {}, None
    for line in lines:
        match = _COORDINATES.match(line)
        if match:
            key = int(match.group(1))
            table[key] = []
        elif "Coordinates for next Run" in line:
            key = "next"
            table[key] = []
        elif len(line.split()) == 3:
            table[key].append([float(field) for field in line.split()])
    return {key: np.array(rows) for key, rows in table.items()}


def _movie(lines) -> dict:
    """``{step: array (atoms, 3)}`` in angstrom."""

    table, key = {}, None
    for line in lines:
        match = _STEP.match(line)
        if match:
            key = int(match.group(1))
            table[key] = []
        elif len(line.split()) == 4:
            table[key].append([float(field) for field in line.split()[1:]])
    return {key: np.array(rows) for key, rows in table.items()}


def _layout(lines) -> list:
    """A file without its numbers: text lines as they are, the length of the others."""

    return [len(line) if _NUMBER.search(line) else line for line in lines]


class _FortranComparison(unittest.TestCase):
    """Comparisons of one file of a run with the Fortran file, step by step."""

    def assert_md_nrg(self, mine: dict, fortran: dict, steps) -> None:
        self.assertEqual(list(mine), list(steps))
        for step in steps:
            ours, theirs = mine[step], fortran[step]
            self.assertEqual(ours[0], theirs[0], msg=f"time of step {step}")
            self.assertEqual(ours[4], theirs[4], msg=f"target temperature of step {step}")
            np.testing.assert_allclose(
                ours[1:4], theirs[1:4], rtol=0, atol=ENERGY_TOLERANCE,
                err_msg=f"energies of step {step}",
            )
            self.assertAlmostEqual(
                ours[5], theirs[5], delta=TEMPERATURE_TOLERANCE,
                msg=f"temperature of step {step}",
            )

    def assert_md_mech(self, mine: dict, fortran: dict, steps) -> None:
        for step in steps:
            ours, theirs = mine[step], fortran[step]
            self.assertEqual(ours.shape, theirs.shape)
            for name, columns, tolerance in (
                ("next positions", slice(0, 3), POSITION_TOLERANCE),
                ("positions", slice(3, 6), POSITION_TOLERANCE),
                ("velocities", slice(6, 9), VELOCITY_TOLERANCE),
                ("accelerations", slice(9, 12), ACCELERATION_TOLERANCE),
                ("movable flags", slice(12, 13), 0.0),
            ):
                np.testing.assert_allclose(
                    ours[:, columns], theirs[:, columns], rtol=0, atol=tolerance,
                    err_msg=f"{name} of step {step}",
                )

    def assert_coordinates(self, mine: dict, fortran: dict, keys, tolerance) -> None:
        self.assertEqual(list(mine), list(keys))
        for key in keys:
            np.testing.assert_allclose(
                mine[key], fortran[key], rtol=0, atol=tolerance,
                err_msg=f"coordinates of step {key}",
            )


class LangevinRunTests(_FortranComparison):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fortran = CASES["langevin"]["fortran"]
        cls.status, cls.files, _archive = _run_case(CASES["langevin"], "--no-archive")

    def test_run_finishes_and_reports_the_input(self) -> None:
        self.assertEqual(self.status, 0)
        self.assertEqual(sorted(self.files), sorted((*FILES, "parsec.out")))
        log = self.files["parsec.out"]
        # the paragraph of usrinputfile.F90
        for line in (
            " Molecular dynamics data:",
            " WARNING: running with user defined RNG seed:        12345",
            " WARNING: running with simple RNG engine ",
            ' "Stairstep" cooling scheme ',
            " Temperature modified after            1  iterations",
            " Initial Temperature =  600.0 [K]",
            " Final Temperature =  400.0 [K]",
            " Time step =    20.00 [au] =  0.000 [ps]",
            " Friction Coefficient =  0.00200 [au]",
            " Large coefficient, doing (MACRO)canonical ensemble",
            " Number of MD iterations =     2",
        ):
            self.assertIn(line, log)
        # one self-consistent calculation per call of moldyn
        self.assertEqual(log.count("Self-consistency convergence achieved."), 3)
        self.assertEqual(log.count(" Molecular dynamics step:"), 3)

    def test_files_have_the_layout_of_the_fortran_files(self) -> None:
        for name in FILES:
            self.assertEqual(_layout(self.files[name]), _layout(self.fortran[name]), msg=name)

    def test_energies_and_temperatures_match_fortran(self) -> None:
        mine = _md_nrg(self.files["md_nrg.dat"])
        self.assert_md_nrg(mine, _md_nrg(self.fortran["md_nrg.dat"]), (0, 1, 2))
        # the stair: 100 K down after every step
        self.assertEqual([mine[step][4] for step in (0, 1, 2)], [600.0, 500.0, 400.0])
        # the run is not a copy of its first structure
        self.assertGreater(abs(mine[2][2] - mine[0][2]), 1.0e-3)

    def test_positions_velocities_and_accelerations_match_fortran(self) -> None:
        mine = _md_mech(self.files["md_mech.dat"])
        self.assertEqual(list(mine), [1, 2, 3])
        self.assert_md_mech(mine, _md_mech(self.fortran["md_mech.dat"]), (1, 2, 3))

    def test_coordinates_match_fortran(self) -> None:
        self.assert_coordinates(
            _atom_cor(self.files["atom.cor"]),
            _atom_cor(self.fortran["atom.cor"]),
            (0, 1, 2, "next"),
            ATOM_COR_TOLERANCE,
        )
        self.assert_coordinates(
            _movie(self.files["movie.xyz"]),
            _movie(self.fortran["movie.xyz"]),
            (0, 1, 2),
            MOVIE_TOLERANCE,
        )

    def test_fixed_atom_stays_where_it_is(self) -> None:
        mechanics = _md_mech(self.files["md_mech.dat"])
        coordinates = _atom_cor(self.files["atom.cor"])
        for step in (1, 2, 3):
            np.testing.assert_array_equal(mechanics[step][:, 12], [1.0, 1.0, 0.0])
            np.testing.assert_array_equal(mechanics[step][2, 6:12], 0.0)
        np.testing.assert_array_equal(coordinates[2][2], coordinates[0][2])
        self.assertGreater(np.abs(coordinates[2][:2] - coordinates[0][:2]).max(), 1.0e-3)


class RestartTests(_FortranComparison):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = CASES["restart"]
        cls.fortran = cls.case["fortran"]
        cls.first = cls.case["first_step"]
        cls.status, cls.files, cls.archive = _run_case(cls.case)

    def test_run_continues_from_the_restart_file(self) -> None:
        self.assertEqual(self.status, 0)
        log = self.files["parsec.out"]
        self.assertIn(
            f" Molecular dynamics restarted from mdinit.dat, step {self.first}", log
        )
        self.assertIn(" WARNING: restarted molecular dynamics run", log)
        self.assertIn(" Small coefficient, doing MICROcanonical ensemble", log)
        self.assertEqual(log.count("Self-consistency convergence achieved."), 2)

    def test_energies_and_temperatures_match_the_unbroken_fortran_run(self) -> None:
        self.assert_md_nrg(
            _md_nrg(self.files["md_nrg.dat"]),
            _md_nrg(self.fortran["md_nrg.dat"]),
            (self.first, self.first + 1),
        )

    def test_positions_velocities_and_accelerations_match_the_unbroken_fortran_run(self) -> None:
        lines = self.files["md_mech.dat"]
        mine = _md_mech(lines)
        self.assertEqual(list(mine), [self.first, self.first + 1, self.first + 2])
        # initial.f90 writes the step it restarts from back, with a list-directed header
        self.assertEqual(lines[5], f" Step #{self.first:12d}")
        self.assertEqual(lines[6:9], self.case["mdinit"][1:])
        self.assert_md_mech(
            mine, _md_mech(self.fortran["md_mech.dat"]), (self.first + 1, self.first + 2)
        )

    def test_coordinates_match_the_unbroken_fortran_run(self) -> None:
        self.assert_coordinates(
            _atom_cor(self.files["atom.cor"]),
            _atom_cor(self.fortran["atom.cor"]),
            (self.first, self.first + 1, "next"),
            ATOM_COR_TOLERANCE,
        )
        self.assert_coordinates(
            _movie(self.files["movie.xyz"]),
            _movie(self.fortran["movie.xyz"]),
            (self.first, self.first + 1),
            MOVIE_TOLERANCE,
        )

    def test_archive_holds_the_trajectory(self) -> None:
        fortran = _md_nrg(self.fortran["md_nrg.dat"])
        steps = (self.first, self.first + 1)
        coordinates = _atom_cor(self.fortran["atom.cor"])
        np.testing.assert_allclose(
            self.archive["dynamics_positions"],
            [coordinates[step] for step in steps],
            rtol=0,
            atol=POSITION_TOLERANCE,
        )
        np.testing.assert_allclose(
            self.archive["dynamics_next_positions"],
            coordinates["next"],
            rtol=0,
            atol=ATOM_COR_TOLERANCE,
        )
        # md_nrg.dat: total energy per atom in eV, 1 Ry = 13.6058 eV in PARSEC
        np.testing.assert_allclose(
            self.archive["dynamics_energies"] * 13.6058 / 3.0,
            [fortran[step][2] for step in steps],
            rtol=0,
            atol=ENERGY_TOLERANCE,
        )
        np.testing.assert_allclose(
            self.archive["dynamics_temperatures"],
            [fortran[step][5] for step in steps],
            rtol=0,
            atol=TEMPERATURE_TOLERANCE,
        )
        # the conserved energy of the two steps, without PARSEC's per-atom sums
        conserved = self.archive["dynamics_conserved_energies"]
        np.testing.assert_allclose(
            conserved,
            self.archive["dynamics_energies"] + self.archive["dynamics_kinetic_energies"],
            rtol=0,
            atol=1.0e-12,
        )
        self.assertEqual(self.archive["forces_total"].shape, (3, 3))
        np.testing.assert_allclose(self.archive["forces_total"].sum(axis=0), 0.0, atol=1e-12)


class InputTests(unittest.TestCase):
    def test_molecular_dynamics_is_not_a_relaxation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            translation = parse_parsec_input(
                _write_case(CASES["langevin"], Path(directory))
            )
        self.assertIsNone(translation.relaxation)
        settings = translation.molecular_dynamics
        self.assertEqual(
            (settings.cooling_method, settings.step_num, settings.stride), ("stair", 2, 1)
        )
        self.assertTrue(settings.use_test_rng)
        self.assertFalse(settings.restart)
        self.assertEqual(translation.movable, (True, True, False))
        self.assertTrue(translation.compute_forces)
        self.assertFalse(translation.use_symmetry)

    def test_restart_without_its_file_is_an_input_error(self) -> None:
        error = StringIO()
        with tempfile.TemporaryDirectory() as directory:
            path = _write_case(CASES["restart"], Path(directory), restart_file=False)
            with redirect_stderr(error):
                status = cli_main([str(path), "--quiet", "--no-archive"])
            written = sorted(item.name for item in Path(directory).iterdir())
        self.assertEqual(status, 2)
        self.assertIn("mdinit.dat", error.getvalue())
        self.assertEqual(written, ["H_POTRE.DAT", "O_POTRE.DAT", "parsec.in"])


class RandomNumberChoiceTests(unittest.TestCase):
    """Which random numbers a run draws, so that it can match a Fortran run."""

    def test_a_seeded_run_draws_the_numbers_of_gfortran(self) -> None:
        import os
        from unittest.mock import patch

        from parsec_python.MolecularDynamics.random_numbers import (
            BufferedRandom,
            GfortranRandom,
            ParsecTestRandom,
        )
        from parsec_python.MolecularDynamics.run import _seeded_compiler, seeded_random_numbers
        from parsec_python.MolecularDynamics.settings import MolecularDynamicsSettings

        def source(keywords, random=None):
            settings = MolecularDynamicsSettings.from_keywords(
                {"molecular_dynamics": "true", **keywords}
            )
            return settings.random_source(random, compiler=_seeded_compiler(settings, random))

        with patch.dict(os.environ):
            os.environ.pop("PARSEC_MD_RANDOM_NUMBERS", None)
            self.assertEqual(seeded_random_numbers(), "gfortran")
            # rng_seed alone: PARSEC seeds the generator of its compiler.
            self.assertIsInstance(source({"rng_seed": "4711"}), GfortranRandom)
            # PARSEC's own generator, the clock, or numbers handed in.
            self.assertIsInstance(source({"rng_seed": "4711", "test_rng": "true"}), ParsecTestRandom)
            self.assertIsInstance(source({}), BufferedRandom)
            self.assertIsInstance(source({"rng_seed": "4711"}, random=3), BufferedRandom)
            os.environ["PARSEC_MD_RANDOM_NUMBERS"] = "numpy"
            self.assertIsInstance(source({"rng_seed": "4711"}), BufferedRandom)
            os.environ["PARSEC_MD_RANDOM_NUMBERS"] = "ifort"
            with self.assertRaisesRegex(ValueError, "gfortran or numpy"):
                source({"rng_seed": "4711"})


if __name__ == "__main__":
    unittest.main()
