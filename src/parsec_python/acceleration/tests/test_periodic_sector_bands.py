"""The band structure of a bulk cell whose SCF solves in the sectors of its point group (``PARSEC_PERIODIC_SECTOR_BANDS``).

Under ``auto`` (the default) the SCF of a bulk cell at the Gamma point that asks for ``bandstruc`` keeps the sectors
of its point group, and the bands read what the SCF leaves on the full grid: the converged effective potential, which
the result of the sectors holds expanded from the orbits, the Fermi level, and the number of states of the full grid
(:func:`parsec_python.acceleration.Output.post_scf.band_hand_over`).  Under ``off`` the whole SCF of such a cell is
solved on the full grid, as before.  The bands, their files and the SCF of the two routes must agree.  After a
relaxation the bands take the projectors of the last structure, as ``calc_bands`` after PARSEC's movement loop.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import parsec_python.acceleration.cli as cli_module
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.tests.test_cli_periodic_sector_forces import (
    SILICON_SETTINGS,
    _Prepared,
    _numbers,
    displaced_positions,
    write_silicon,
)

#: A path of the band structure with three points from Gamma to X (reciprocal-lattice coordinates).
BANDS = "begin bandstruc\n1 0.0 0.0 0.0  0.5 0.0 0.0  G-X\nend bandstruc\nbandstruc_points 3\n"
FORCES = "Skip_force: .false.\n"
#: The perfect diamond cube of ``examples/3d_Si``: Td about the atom at the origin with the face-centring
#: translations (the weighted sectors of the cell).
PERFECT = displaced_positions(0.0)


class CommandLineRoutingTests(unittest.TestCase):
    """What symmetry the command line prepares a cell with a band structure with (the preparation itself is not run)."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        self.count = 0

    def route(self, extra: str, options=(), environment=None):
        """``("prepared", symmetry)`` or ``("exit", status, stderr)`` for the input with ``extra``."""

        self.count += 1
        path = write_silicon(self.directory / f"case{self.count}", extra, positions=PERFECT)
        seen = {}

        def prepare(problem, **keywords):
            seen.update(keywords)
            raise _Prepared

        errors = StringIO()
        variables = {
            "PARSEC_PERIODIC_SECTOR_FORCES": "auto",
            "PARSEC_PERIODIC_SECTOR_BANDS": "auto",
            **(environment or {}),
        }
        with mock.patch.dict(os.environ, variables), mock.patch.object(
            cli_module, "prepare_single_point", side_effect=prepare
        ), redirect_stdout(StringIO()), redirect_stderr(errors):
            try:
                status = cli_module.main([str(path), "--quiet", "--no-archive", "--device", "cpu", *options])
            except _Prepared:
                return ("prepared", seen["symmetry"])
        return ("exit", status, errors.getvalue())

    def test_a_band_structure_keeps_the_sectors_of_the_scf(self) -> None:
        self.assertEqual(self.route(BANDS), ("prepared", "auto"))
        # with forces, whose sectors PARSEC_PERIODIC_SECTOR_FORCES=auto keeps
        self.assertEqual(self.route(BANDS + FORCES), ("prepared", "auto"))
        self.assertEqual(self.route(BANDS, ("--symmetry", "on")), ("prepared", "on"))
        self.assertEqual(self.route(BANDS, ("--symmetry", "off")), ("prepared", "off"))
        self.assertEqual(self.route("Ignore_Symmetry: .true.\n" + BANDS), ("prepared", "off"))
        # forces on the full grid by their own switch
        self.assertEqual(
            self.route(BANDS + FORCES, environment={"PARSEC_PERIODIC_SECTOR_FORCES": "off"}), ("prepared", "off")
        )

    def test_the_switch_off_keeps_the_full_grid_as_before(self) -> None:
        off = {"PARSEC_PERIODIC_SECTOR_BANDS": "off"}
        self.assertEqual(self.route(BANDS, environment=off), ("prepared", "off"))
        self.assertEqual(self.route(BANDS + FORCES, environment=off), ("prepared", "off"))
        outcome = self.route(BANDS, ("--symmetry", "on"), off)
        self.assertEqual(outcome[:2], ("exit", 2), outcome)
        self.assertIn(
            "--symmetry on cannot be used with a band structure of a periodic cell, which the point-group sectors of "
            "a cell do not compute",
            outcome[2],
        )
        # a single point without bands, as before
        self.assertEqual(self.route("", environment=off), ("prepared", "auto"))
        outcome = self.route(BANDS, environment={"PARSEC_PERIODIC_SECTOR_BANDS": "sometimes"})
        self.assertEqual(outcome[:2], ("exit", 2))
        self.assertIn("PARSEC_PERIODIC_SECTOR_BANDS must be auto or off", outcome[2])

    def test_the_refusal_of_the_forces_names_the_bands_only_under_off(self) -> None:
        problem = SimpleNamespace(
            periodic_cell=object(),
            grid=SimpleNamespace(fortran_routines=False),
            point_charges=(),
            bands=object(),
            has_kpoint_sampling=False,
        )
        translation = SimpleNamespace(
            problem=problem,
            molecular_dynamics=None,
            relaxation=None,
            compute_forces=True,
            routines=SimpleNamespace(fortran=False),
            movable=[True, True],
            polarizability_field=None,
            nscf=None,
            restart_run=False,
            restart_from_wfndat=False,
        )
        refusal = cli_module.periodic_sector_forces_refusal
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SECTOR_FORCES": "auto", "PARSEC_PERIODIC_SECTOR_BANDS": "auto"}):
            self.assertIsNone(refusal(translation))
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SECTOR_FORCES": "auto", "PARSEC_PERIODIC_SECTOR_BANDS": "off"}):
            self.assertEqual(refusal(translation), "a band structure")


class BandHandOverTests(unittest.TestCase):
    """:func:`band_hand_over` cuts the level list of weighted sectors to the states of the full grid, nothing else."""

    @staticmethod
    def system(states=None, electrons=8.0):
        scf = SimpleNamespace(number_of_states=states, fermi_temperature_kelvin=300.0, fixed_occupations=None)
        reference = SimpleNamespace(
            input=SimpleNamespace(scf=scf, eigensolver=SimpleNamespace(subspace_buffer=6)),
            electron_count=electrons,
            grid=SimpleNamespace(size=1000),
        )
        return SimpleNamespace(reference=reference)

    def test_a_full_grid_result_is_handed_over_as_it_is(self) -> None:
        from parsec_python.acceleration.Output.post_scf import band_hand_over

        result = SimpleNamespace(eigenvalues=np.arange(6.0), fermi_level=0.5, k_points=None)
        self.assertIs(band_hand_over(self.system(6), result), result)
        # the default count of the full grid: ceil(N_e / 2) + 6
        default = SimpleNamespace(eigenvalues=np.arange(10.0), fermi_level=0.5, k_points=None)
        self.assertIs(band_hand_over(self.system(None), default), default)

    def test_the_partners_past_states_num_are_cut(self) -> None:
        from parsec_python.acceleration.Output.post_scf import band_hand_over

        potential = np.ones(1000)
        # a triplet completes the multiplet of the sixth level: eight listed levels
        result = SimpleNamespace(
            eigenvalues=np.array([-1.0, -0.5, -0.5, -0.5, 0.1, 0.2, 0.2, 0.2]),
            fermi_level=0.05,
            k_points=None,
            converged=True,
            output_effective_potential=potential,
        )
        handed = band_hand_over(self.system(6), result)
        np.testing.assert_array_equal(handed.eigenvalues, [-1.0, -0.5, -0.5, -0.5, 0.1, 0.2])
        self.assertIs(handed.output_effective_potential, potential)
        self.assertTrue(handed.converged)
        # the Fermi level of the six levels, as the SCF loop finds it on the full grid
        from parsec_python.Occupations import fermi_occupations

        expected = fermi_occupations(np.array([-1.0, -0.5, -0.5, -0.5, 0.1, 0.2]), 8.0, 300.0).fermi_level
        self.assertEqual(handed.fermi_level, expected)
        self.assertNotEqual(handed.fermi_level, 0.05)


class BandsOfTheLastStructureTests(unittest.TestCase):
    """Both command lines hand the bands the system of the last structure of a relaxation (a small cell on the CPU)."""

    def test_the_band_stage_reads_the_moved_atoms(self) -> None:
        from parsec_python import cli as reference_cli
        from parsec_python.acceleration.tests.test_post_scf_outputs import BANDS as SHORT_PATH, CELL, _write

        relaxation = "Minimization: bfgs\nMovement_Num: 1\nForce_Min: 1.0e-5\n"
        original = reference_cli._run_band_structure
        for label, main, options in (
            ("reference", reference_cli.main, ()),
            ("accelerated", cli_module.main, ("--device", "cpu")),
        ):
            with self.subTest(command_line=label), tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
                directory = Path(name)
                path = _write(directory, CELL, SHORT_PATH + relaxation)
                handed = []

                def record(system, *arguments, **keywords):
                    handed.append(np.array([atom.position for atom in system.atoms]))
                    return original(system, *arguments, **keywords)

                errors = StringIO()
                with mock.patch.object(reference_cli, "_run_band_structure", side_effect=record), redirect_stdout(
                    StringIO()
                ), redirect_stderr(errors):
                    status = main([str(path), "--quiet", *options])
                self.assertIn(status, (0, 4), errors.getvalue())
                if label == "accelerated":
                    self.assertNotIn("running the reference parsec_python.cli solver", errors.getvalue())
                with np.load(directory / "parsec_python_results.npz", allow_pickle=True) as data:
                    computed = np.asarray(data["relaxation_positions"])
                self.assertEqual(computed.shape[0], 2)
                self.assertGreater(float(np.abs(computed[-1] - computed[0]).max()), 1.0e-4)
                self.assertEqual(len(handed), 1)
                np.testing.assert_allclose(handed[0], computed[-1], rtol=0, atol=1e-12)
                self.assertTrue((directory / "bands.dat").is_file())


def _run(directory: Path, extra: str, setting: str, *, positions=None, settings=None):
    """The accelerated command line on the silicon cube: ``(status, archive, bands archive, parsec.out, stderr)``."""

    path = write_silicon(directory, extra, positions=PERFECT if positions is None else positions, settings=settings)
    errors = StringIO()
    variables = {"PARSEC_PERIODIC_SECTOR_BANDS": setting, "PARSEC_PERIODIC_SECTOR_FORCES": "auto"}
    with mock.patch.dict(os.environ, variables), redirect_stdout(StringIO()), redirect_stderr(errors):
        status = cli_module.main([str(path), "--quiet", "--backend", "cupy"])

    def archive(name):
        if not (directory / name).is_file():
            return {}
        with np.load(directory / name, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}

    log = (directory / "parsec.out").read_text(encoding="utf-8") if (directory / "parsec.out").is_file() else ""
    return status, archive("parsec_python_results.npz"), archive("parsec_python_bands.npz"), log, errors.getvalue()


#: 22 states: the level list of the weighted sectors holds exactly 22 states (no multiplet cut).
BAND_SETTINGS = {**SILICON_SETTINGS, "States_Num": "22"}
#: 21 states cut a triplet: the sectors list 22 levels, the full grid 21.
CUT_SETTINGS = {**SILICON_SETTINGS, "States_Num": "21"}
SECTORS_LINE = "exact group of 96 operations detected"


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class SectorBandStructureTests(unittest.TestCase):
    """The perfect silicon cube with a band path: SCF in the weighted sectors + bands against the full grid."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._directory.name)
        cls.full = _run(cls.root / "full", BANDS, "off", settings=BAND_SETTINGS)
        cls.sectors = _run(cls.root / "sectors", BANDS, "auto", settings=BAND_SETTINGS)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_the_bands_of_the_sectors_equal_those_of_the_full_grid(self) -> None:
        for label, (status, _, _, log, errors) in (("full", self.full), ("sectors", self.sectors)):
            self.assertEqual(status, 0, f"{label}: {errors}\n{log[-3000:]}")
        _, full, full_bands, full_log, _ = self.full
        _, sectors, sector_bands, sector_log, _ = self.sectors
        # the SCF of the sectors is weighted (96 operations, sectors of a subgroup of 16); the full grid's has none
        self.assertIn(SECTORS_LINE, sector_log)
        self.assertNotIn(SECTORS_LINE, full_log)
        self.assertIn("Band structure (non-self-consistent, Fortran calc_bands)", sector_log)
        self.assertIn("Band structure (non-self-consistent, Fortran calc_bands)", full_log)
        shape = np.asarray(full_bands["eigenvalues_ry"]).shape
        self.assertEqual(shape[-1], 22)
        self.assertEqual(np.asarray(sector_bands["eigenvalues_ry"]).shape, shape)
        difference = float(np.abs(sector_bands["eigenvalues_ry"] - full_bands["eigenvalues_ry"]).max())
        print(f"\n largest band difference sectors - full grid: {difference:.3e} Ry")
        np.testing.assert_allclose(sector_bands["eigenvalues_ry"], full_bands["eigenvalues_ry"], rtol=0, atol=1e-8)
        self.assertAlmostEqual(float(sector_bands["fermi_level_ry"]), float(full_bands["fermi_level_ry"]), delta=1e-8)
        self.assertAlmostEqual(float(sectors["energy_total_ry"]), float(full["energy_total_ry"]), delta=1e-8)
        np.testing.assert_allclose(
            sectors["output_effective_potential_ry"], full["output_effective_potential_ry"], rtol=0, atol=1e-8
        )

    def test_the_band_files_equal_those_of_the_full_grid(self) -> None:
        for name in ("bands.dat", "bands_plot.dat"):
            with self.subTest(file=name):
                full, full_text = _numbers(self.root / "full" / name)
                sectors, sector_text = _numbers(self.root / "sectors" / name)
                self.assertEqual(sector_text, full_text)
                # the numbers as printed (round-off of the last digit)
                np.testing.assert_allclose(sectors, full, rtol=0, atol=1e-6)


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class CutMultipletTests(unittest.TestCase):
    """States_Num 21 cuts a triplet: the sectors list its 22 partners, the bands still get 21 states per k-point and
    the Fermi level of the 21 levels of the full grid.  (The full grid, which solves two of the three partners,
    stalls at a residual of about 2e-8 and ends unconverged after Max_Iter, as before; the sectors converge.)"""

    def test_the_bands_get_the_states_and_the_fermi_level_of_the_full_grid(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            full = _run(root / "full", BANDS, "off", settings=CUT_SETTINGS)
            sectors = _run(root / "sectors", BANDS, "auto", settings=CUT_SETTINGS)
        self.assertIn(full[0], (0, 3), f"{full[4]}\n{full[3][-3000:]}")
        self.assertEqual(sectors[0], 0, f"{sectors[4]}\n{sectors[3][-3000:]}")
        self.assertIn(SECTORS_LINE, sectors[3])
        self.assertEqual(np.asarray(full[1]["eigenvalues_ry"]).size, 21)
        self.assertEqual(np.asarray(sectors[1]["eigenvalues_ry"]).size, 22)
        shape = np.asarray(full[2]["eigenvalues_ry"]).shape
        self.assertEqual(shape[-1], 21)
        self.assertEqual(np.asarray(sectors[2]["eigenvalues_ry"]).shape, shape)
        np.testing.assert_allclose(sectors[2]["eigenvalues_ry"], full[2]["eigenvalues_ry"], rtol=0, atol=1e-7)
        # the SCF of the sectors finds another level in the gap (its bisection brackets 22 levels)
        self.assertGreater(abs(float(sectors[1]["fermi_level_ry"]) - float(full[1]["fermi_level_ry"])), 1e-6)
        self.assertAlmostEqual(float(sectors[2]["fermi_level_ry"]), float(full[2]["fermi_level_ry"]), delta=1e-7)


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class BandsAfterRelaxationTests(unittest.TestCase):
    """After a relaxation the bands take the last structure (its projectors), in the sectors as on the full grid."""

    def test_the_bands_take_the_system_of_the_last_structure(self) -> None:
        relaxation = "Minimization: bfgs\nMovement_Num: 1\nForce_Min: 1.0e-5\n"
        handed = []
        original = cli_module.run_post_scf_outputs

        def record(system, *arguments, **keywords):
            handed.append(np.array([atom.position for atom in system.reference.atoms]))
            return original(system, *arguments, **keywords)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            results = {}
            for setting in ("off", "auto"):
                with mock.patch.object(cli_module, "run_post_scf_outputs", side_effect=record):
                    results[setting] = _run(root / setting, BANDS + relaxation, setting, positions=displaced_positions())
        for setting, (status, _, _, log, errors) in results.items():
            self.assertIn(status, (0, 4), f"{setting}: {errors}\n{log[-3000:]}")
        for setting, positions in zip(("off", "auto"), handed):
            archive = results[setting][1]
            computed = np.asarray(archive["relaxation_positions"])
            self.assertEqual(computed.shape[0], 2)
            self.assertGreater(float(np.abs(computed[-1] - computed[0]).max()), 1.0e-4)
            # the structure of the last SCF, not the input
            np.testing.assert_allclose(positions, computed[-1], rtol=0, atol=1e-12)
        self.assertIn("averaged over 24 operations", results["auto"][3])
        np.testing.assert_allclose(
            results["auto"][2]["eigenvalues_ry"], results["off"][2]["eigenvalues_ry"], rtol=0, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
