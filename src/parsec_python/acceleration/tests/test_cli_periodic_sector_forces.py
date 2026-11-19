"""The command line of the forces of a bulk cell in the sectors of its point group (``PARSEC_PERIODIC_SECTOR_FORCES``).

Under ``auto`` (the default) the single-point forces and the relaxations by an optimizer of a bulk cell at the Gamma
point on the default routines keep the sectors of its point group about the origin of the cell, which the driver
finds (``PARSEC_PERIODIC_SYMMETRY``); the forces are averaged over the full group ``G`` of the sectors, and so are the
forces and moved positions of the ionic loop.  Molecular dynamics, manual movements, ``Move_Flag`` with fixed atoms
and the Fortran-derived routines stay on the full grid, as everything does under ``off``.  The forces of the sectors
must be those of the full grid.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

import parsec_python.acceleration.cli as cli_module
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available

PERIODIC_SILICON = Path(__file__).resolve().parents[3].parent / "examples" / "3d_Si" / "parsec.in"

#: The settings of ``test_periodic_point_group.silicon("pbe")``: SCF to 1e-9 on a grid of 18 points per side.
SILICON_SETTINGS = {
    "Grid_Spacing": "0.57",
    "Expansion_Order": "8",
    "States_Num": "22",
    "Convergence_Criterion": "1.0e-9",
    "Diag_Tolerance": "1.0e-10",
    "Correlation_Type": "pbe",
}


def displaced_positions(distance: float = 0.04, origin_shift: float = 0.0) -> np.ndarray:
    """The atoms of ``examples/3d_Si`` with the four neighbours of the atom at the origin moved ``distance`` bohr
    outward along their bonds (``test_periodic_point_group.displaced_silicon``: Td about the origin, no face-centring
    translations); ``origin_shift`` moves the atom at the origin along z as well (C2v about the origin, Abelian)."""

    side = 10.26
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 5.13, 5.13],
            [5.13, 0.0, 5.13],
            [5.13, 5.13, 0.0],
            [2.565, 2.565, 2.565],
            [2.565, 7.695, 7.695],
            [7.695, 2.565, 7.695],
            [7.695, 7.695, 2.565],
        ]
    )
    positions = []
    for position in base:
        bond = position - side * np.round(position / side)
        if 0.0 < np.linalg.norm(bond) < 0.5 * side * np.sqrt(3.0) / 2.0 + 1.0e-6 and np.allclose(
            np.abs(bond), np.abs(bond[0])
        ):
            position = position + distance * bond / np.linalg.norm(bond)
        positions.append(position)
    positions = np.array(positions)
    positions[0, 2] += origin_shift
    return positions


def write_silicon(directory: Path, extra: str = "", *, positions: np.ndarray | None = None, settings=None) -> Path:
    """``parsec.in`` of the silicon cube in ``directory`` with ``positions`` (default :func:`displaced_positions`),
    the keyword values of ``settings`` (default :data:`SILICON_SETTINGS`), ``Skip_force`` removed and ``extra``
    appended."""

    positions = displaced_positions() if positions is None else positions
    changes = SILICON_SETTINGS if settings is None else settings
    lines, in_coordinates, row = [], False, 0
    for line in PERIODIC_SILICON.read_text().splitlines():
        stripped = line.strip().lower()
        keyword = line.split(":", 1)[0].strip()
        if stripped.startswith("begin atom_coord"):
            in_coordinates = True
            lines.append(line)
            continue
        if stripped.startswith("end atom_coord"):
            in_coordinates = False
        elif in_coordinates:
            lines.append("    " + "  ".join(repr(float(value)) for value in positions[row]))
            row += 1
            continue
        if keyword.lower() == "skip_force":
            continue
        lines.append(f" {keyword}: {changes[keyword]}" if ":" in line and keyword in changes else line)
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(PERIODIC_SILICON.parent / "Si_POTRE.DAT", directory / "Si_POTRE.DAT")
    path = directory / "parsec.in"
    path.write_text("\n".join(lines) + "\n" + extra, encoding="utf-8")
    return path


class _Prepared(BaseException):
    """Raised by the stand-in of ``prepare_single_point``: the routing is the subject, not the calculation (not an
    ``Exception``, which the command line reports as a failed calculation)."""


FORCES = "Skip_force: .false.\n"
BFGS = "Minimization: bfgs\nMovement_Num: 2\n"
SIMPLE = "Minimization: simple\nMovement_Num: 2\n"
MANUAL = "Minimization: manual\nMovement_Num: 0\n"
DYNAMICS = "Molecular_Dynamics: .true.\nStep_num: 1\nTinit: 300.0\nT_Final: 300.0\nT_Step: 0.0\nTime_Step: 20.0\n"


class CommandLineRoutingTests(unittest.TestCase):
    """What symmetry the command line prepares a cell with (the preparation itself is not run)."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        self.count = 0

    def route(self, extra: str, options=(), environment=None, *, text_of=None):
        """``("prepared", symmetry)`` or ``("exit", status, stderr)`` for the input with ``extra``."""

        self.count += 1
        path = write_silicon(self.directory / f"case{self.count}", extra, **(text_of or {}))
        seen = {}

        def prepare(problem, **keywords):
            seen.update(keywords)
            raise _Prepared

        errors = StringIO()
        variables = {"PARSEC_PERIODIC_SECTOR_FORCES": "auto", **(environment or {})}
        with mock.patch.dict(os.environ, variables), mock.patch.object(
            cli_module, "prepare_single_point", side_effect=prepare
        ), redirect_stdout(StringIO()), redirect_stderr(errors):
            try:
                status = cli_module.main([str(path), "--quiet", "--no-archive", "--device", "cpu", *options])
            except _Prepared:
                return ("prepared", seen["symmetry"])
        return ("exit", status, errors.getvalue())

    def test_forces_and_relaxations_keep_the_sectors(self) -> None:
        for extra in (FORCES, BFGS, SIMPLE, FORCES + BFGS):
            with self.subTest(extra=extra):
                self.assertEqual(self.route(extra), ("prepared", "auto"))
        self.assertEqual(self.route(FORCES, ("--symmetry", "on")), ("prepared", "on"))
        self.assertEqual(self.route(BFGS, ("--symmetry", "on")), ("prepared", "on"))
        self.assertEqual(self.route(FORCES, ("--symmetry", "off")), ("prepared", "off"))
        # a single point without forces, as before
        self.assertEqual(self.route(""), ("prepared", "auto"))

    def test_the_switch_off_keeps_the_full_grid_as_before(self) -> None:
        off = {"PARSEC_PERIODIC_SECTOR_FORCES": "off"}
        for extra in (FORCES, BFGS, DYNAMICS):
            with self.subTest(extra=extra):
                self.assertEqual(self.route(extra, environment=off), ("prepared", "off"))
        outcome = self.route(FORCES, ("--symmetry", "on"), off)
        self.assertEqual(outcome[:2], ("exit", 2))
        self.assertIn(
            "--symmetry on cannot be used with forces of a periodic cell, which the point-group sectors of a cell "
            "do not compute",
            outcome[2],
        )
        self.assertEqual(self.route("", environment=off), ("prepared", "auto"))
        outcome = self.route(FORCES, environment={"PARSEC_PERIODIC_SECTOR_FORCES": "sometimes"})
        self.assertEqual(outcome[:2], ("exit", 2))
        self.assertIn("PARSEC_PERIODIC_SECTOR_FORCES must be auto or off", outcome[2])

    def test_dynamics_manual_moves_fixed_atoms_and_fortran_routines_keep_the_full_grid(self) -> None:
        fixed = (
            BFGS.replace("Minimization", "Move_Flag: 4\nMinimization"),
            "Move_Flag fixes atoms",
        )
        for extra, reason in (
            (DYNAMICS, "molecular dynamics"),
            (MANUAL, "manual movements"),
            fixed,
            ("Use_Fortran_Routines: .true.\n" + FORCES, "Use_Fortran_Routines"),
            ("Use_Fortran_Routines: .true.\n" + BFGS, "Use_Fortran_Routines"),
        ):
            with self.subTest(extra=extra):
                self.assertEqual(self.route(extra), ("prepared", "off"))
                outcome = self.route(extra, ("--symmetry", "on"))
                self.assertEqual(outcome[:2], ("exit", 2), outcome)
                self.assertIn("--symmetry on cannot be used with", outcome[2])
                self.assertIn(reason, outcome[2])

    def test_ignore_symmetry_keeps_the_full_grid_unless_the_command_line_asks(self) -> None:
        self.assertEqual(self.route("Ignore_Symmetry: .true.\n" + FORCES), ("prepared", "off"))
        self.assertEqual(self.route("Ignore_Symmetry: .true.\n" + FORCES, ("--symmetry", "on")), ("prepared", "on"))

    def test_the_refusal_names_what_keeps_the_full_grid(self) -> None:
        from types import SimpleNamespace

        def translation(**fields):
            problem = SimpleNamespace(
                periodic_cell=object(),
                grid=SimpleNamespace(fortran_routines=False),
                point_charges=(),
                bands=None,
                has_kpoint_sampling=False,
            )
            base = dict(
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
            base.update(fields)
            return SimpleNamespace(**base)

        refusal = cli_module.periodic_sector_forces_refusal
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SECTOR_FORCES": "auto"}):
            self.assertIsNone(refusal(translation()))
            self.assertIsNone(refusal(translation(relaxation=SimpleNamespace(method="bfgs"))))
            self.assertIn("supercell", refusal(translation(), supercell=True))
            self.assertIn("no forces", refusal(translation(compute_forces=False)))
            self.assertIn("Move_Flag", refusal(translation(movable=[True, False])))
            self.assertIn("manual", refusal(translation(relaxation=SimpleNamespace(method="manual"))))
            self.assertIn("molecular dynamics", refusal(translation(molecular_dynamics=object())))
            self.assertIn("Restart_From_Wfndat", refusal(translation(restart_run=True, restart_from_wfndat=True)))
            self.assertIn("non-self-consistent", refusal(translation(nscf=object())))
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SECTOR_FORCES": "off"}):
            self.assertEqual(refusal(translation()), "PARSEC_PERIODIC_SECTOR_FORCES=off")


def _run(directory: Path, extra: str, setting: str, *, positions=None, options=(), settings=None):
    """The accelerated command line on the silicon cube: ``(status, archive, parsec.out, stderr)``."""

    path = write_silicon(directory, extra, positions=positions, settings=settings)
    errors = StringIO()
    with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SECTOR_FORCES": setting}), redirect_stdout(
        StringIO()
    ), redirect_stderr(errors):
        status = cli_module.main([str(path), "--quiet", "--backend", "cupy", *options])
    archive = {}
    if (directory / "parsec_python_results.npz").is_file():
        with np.load(directory / "parsec_python_results.npz", allow_pickle=True) as data:
            archive = {key: data[key] for key in data.files}
    log = (directory / "parsec.out").read_text(encoding="utf-8") if (directory / "parsec.out").is_file() else ""
    return status, archive, log, errors.getvalue()


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class CommandLineSectorForceTests(unittest.TestCase):
    """The forces of the displaced silicon cube through the command line, in sectors and on the full grid."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(cls._directory.name)
        cls.full = _run(root / "full", FORCES, "off")
        cls.sectors = _run(root / "sectors", FORCES, "auto")
        # a relaxation without movements: the forces of the ionic loop with the group of the sectors
        cls.relaxed = _run(root / "relaxation", "Minimization: bfgs\nMovement_Num: 0\n", "auto")
        cls.required = _run(root / "required", FORCES, "auto", options=("--symmetry", "on"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_forces_in_the_sectors_equal_those_of_the_full_grid(self) -> None:
        for label, (status, archive, log, errors) in (("full", self.full), ("sectors", self.sectors)):
            self.assertEqual(status, 0, f"{label}: {errors}\n{log[-3000:]}")
        _, full, full_log, _ = self.full
        _, sectors, sector_log, _ = self.sectors
        self.assertIn("skipped: periodic cell", full_log)
        self.assertNotIn("Forces of the periodic cell in the sectors", full_log)
        self.assertIn("weights 1, 1, 3, 3", sector_log)
        self.assertIn("Forces of the periodic cell in the sectors of its point group: averaged over 24 operations", sector_log)
        # one atom per orbit of the group of the sectors keeps its projectors
        self.assertIn("projectors of ", sector_log)
        self.assertGreater(float(np.abs(full["forces_total"]).max()), 1.0e-3)
        np.testing.assert_allclose(sectors["forces_total"], full["forces_total"], rtol=0, atol=2e-6)
        self.assertAlmostEqual(float(sectors["energy_total_ry"]), float(full["energy_total_ry"]), delta=1e-7)

    def test_symmetry_on_is_accepted_with_forces(self) -> None:
        status, archive, log, errors = self.required
        self.assertEqual(status, 0, f"{errors}\n{log[-3000:]}")
        self.assertIn("averaged over 24 operations", log)
        np.testing.assert_allclose(archive["forces_total"], self.full[1]["forces_total"], rtol=0, atol=2e-6)

    def test_the_ionic_loop_takes_the_group_of_the_sectors(self) -> None:
        status, archive, log, errors = self.relaxed
        self.assertIn(status, (0, 4), f"{errors}\n{log[-3000:]}")
        self.assertIn("averaged over 24 operations", log)
        forces = np.asarray(archive["relaxation_forces"])[0]
        np.testing.assert_allclose(forces, self.full[1]["forces_total"], rtol=0, atol=2e-6)

    def test_a_cell_in_abelian_sectors_has_the_forces_of_the_full_grid(self) -> None:
        # the atom at the origin moved along z as well: C2v about the origin, Abelian, every level solved
        positions = displaced_positions(origin_shift=0.05)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            full = _run(root / "full", FORCES, "off", positions=positions)
            sectors = _run(root / "sectors", FORCES, "auto", positions=positions)
        for label, (status, _, log, errors) in (("full", full), ("sectors", sectors)):
            self.assertEqual(status, 0, f"{label}: {errors}\n{log[-3000:]}")
        self.assertIn("Forces of the periodic cell in the sectors of its point group", sectors[2])
        self.assertGreater(float(np.abs(full[1]["forces_total"]).max()), 1.0e-3)
        np.testing.assert_allclose(sectors[1]["forces_total"], full[1]["forces_total"], rtol=0, atol=2e-6)

    def test_molecular_dynamics_reports_the_full_grid(self) -> None:
        # Eigensolver: none (PARSEC's TEST mode) reports the setup and ends before the SCF
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            status, _, log, errors = _run(Path(name), DYNAMICS + "Eigensolver: none\n", "auto")
        self.assertEqual(status, 0, errors)
        self.assertIn("skipped: periodic cell", log)
        self.assertNotIn("Forces of the periodic cell in the sectors", log)


_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?")


def _numbers(path: Path) -> tuple[np.ndarray, str]:
    """The numbers of a text file and its words with each number replaced by ``#``: the layout of the numbers (a
    ``-0.000000`` of round-off against a ``0.000000``, a list of zeros on one line) is not compared."""

    text = path.read_text(encoding="utf-8", errors="replace")
    values = [float(value.replace("D", "e").replace("d", "e")) for value in _NUMBER.findall(text)]
    return np.array(values), " ".join(_NUMBER.sub("#", text).split())


#: Two BFGS movements from the displaced cube (largest force 5e-3 Ry/bohr, below the default Force_Min).
RELAXATION = "Minimization: bfgs\nMovement_Num: 2\nForce_Min: 1.0e-5\n"
#: One more movement from the relax_restart.dat of :data:`RELAXATION`.
RESTARTED = "Minimization: bfgs\nMovement_Num: 1\nForce_Min: 1.0e-5\n"
RESTART_SETTINGS = {**SILICON_SETTINGS, "relax_restart": ".true."}
STEEPEST_DESCENT = "Minimization: simple\nMovement_Num: 1\nForce_Min: 1.0e-5\n"
#: One step of dynamics with velocities of a fixed seed, which break the point group.
SEEDED_DYNAMICS = (
    "Molecular_Dynamics: .true.\nTinit: 300\nT_Final: 300\nT_Step: 0\nTime_Step: 20\nStep_Num: 1\nRng_Seed: 7\n"
)


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class CommandLineSectorRelaxationTests(unittest.TestCase):
    """Relaxations of the displaced silicon cube with movements in the sectors (BFGS, its restart, an Abelian cell,
    steepest descent) equal those of the full grid; its dynamics stays on the full grid."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._directory.name)
        cls.full = _run(cls.root / "full", RELAXATION, "off")
        cls.sectors = _run(cls.root / "sectors", RELAXATION, "auto")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_the_relaxation_in_the_sectors_moves_as_on_the_full_grid(self) -> None:
        for label, (status, _, log, errors) in (("full", self.full), ("sectors", self.sectors)):
            self.assertIn(status, (0, 4), f"{label}: {errors}\n{log[-3000:]}")
        _, full, full_log, _ = self.full
        _, sectors, sector_log, _ = self.sectors
        self.assertNotIn("Forces of the periodic cell in the sectors", full_log)
        self.assertIn(
            "Forces of the periodic cell in the sectors of its point group: averaged over 24 operations; projectors of "
            "4 of 8 atoms",
            sector_log,
        )
        # three structures: the input and two movements, which moved the atoms
        positions = np.asarray(sectors["relaxation_positions"])
        self.assertEqual(positions.shape, (3, 8, 3))
        self.assertGreater(float(np.abs(positions[-1] - positions[0]).max()), 1.0e-3)
        np.testing.assert_allclose(positions, full["relaxation_positions"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(sectors["relaxation_energies"], full["relaxation_energies"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(sectors["relaxation_forces"], full["relaxation_forces"], rtol=0, atol=2e-6)
        np.testing.assert_allclose(
            sectors["relaxation_final_positions"], full["relaxation_final_positions"], rtol=0, atol=1e-6
        )
        # every structure has its forces averaged over the group; the summary after the movements is that of the
        # full grid
        self.assertEqual(sector_log.count("symmetrizing forces according to symmetry operations"), 3)
        self.assertEqual(self._summary(sector_log), self._summary(full_log))
        self.assertIn(" Structures computed            :        3", self._summary(sector_log))

    @staticmethod
    def _summary(log: str) -> list[str]:
        """The lines of the summary of a relaxation in parsec.out."""

        starts = (" Structures computed", " Total energy of the last one", " Largest force on a movable atom")
        return [line for line in log.splitlines() if line.startswith(starts) or "Minimum force is" in line]

    def test_the_files_of_the_relaxation_equal_those_of_the_full_grid(self) -> None:
        for name in ("atom.cor", "relax_restart.dat"):
            with self.subTest(file=name):
                full, full_text = _numbers(self.root / "full" / name)
                sectors, sector_text = _numbers(self.root / "sectors" / name)
                self.assertEqual(sector_text, full_text)
                self.assertGreater(full.size, 100)
                np.testing.assert_allclose(sectors, full, rtol=0, atol=1e-6)

    def test_a_restart_continues_in_the_sectors(self) -> None:
        # the relax_restart.dat of the relaxation in the sectors (on the group to round-off) continued in the
        # sectors and on the full grid
        restarted = {}
        for setting in ("off", "auto"):
            directory = self.root / f"restart_{setting}"
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.root / "sectors" / "relax_restart.dat", directory / "relax_restart.dat")
            restarted[setting] = _run(directory, RESTARTED, setting, settings=RESTART_SETTINGS)
            status, _, log, errors = restarted[setting]
            self.assertIn(status, (0, 4), f"{setting}: {errors}\n{log[-3000:]}")
            self.assertIn("coordinates in relax_restart.dat", log)
        full, sectors = restarted["off"][1], restarted["auto"][1]
        self.assertIn("averaged over 24 operations", restarted["auto"][2])
        np.testing.assert_allclose(
            sectors["relaxation_positions"][0], self.sectors[1]["relaxation_final_positions"], rtol=0, atol=1e-9
        )
        np.testing.assert_allclose(sectors["relaxation_positions"], full["relaxation_positions"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(sectors["relaxation_energies"], full["relaxation_energies"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(sectors["relaxation_forces"], full["relaxation_forces"], rtol=0, atol=2e-6)

    def test_a_relaxation_in_abelian_sectors_moves_as_on_the_full_grid(self) -> None:
        # C2v about the origin: commuting signed permutations, every level solved, no average needed
        positions = displaced_positions(origin_shift=0.05)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            full = _run(root / "full", RELAXATION, "off", positions=positions)
            sectors = _run(root / "sectors", RELAXATION, "auto", positions=positions)
            for label, (status, _, log, errors) in (("full", full), ("sectors", sectors)):
                self.assertIn(status, (0, 4), f"{label}: {errors}\n{log[-3000:]}")
            self.assertIn("an Abelian group of 4 operations", sectors[2])
            self.assertEqual(np.asarray(sectors[1]["relaxation_positions"]).shape, (3, 8, 3))
            for key in ("relaxation_positions", "relaxation_final_positions", "relaxation_energies"):
                np.testing.assert_allclose(sectors[1][key], full[1][key], rtol=0, atol=1e-6, err_msg=key)
            for name in ("atom.cor", "relax_restart.dat"):
                np.testing.assert_allclose(
                    _numbers(root / "sectors" / name)[0], _numbers(root / "full" / name)[0], rtol=0, atol=1e-6
                )

    def test_steepest_descent_keeps_the_atoms_the_group_fixes(self) -> None:
        # The simple scheme of domove.f90 moves every nonzero force component by about Min_Step, so the forces of
        # round-off on the four atoms that Td fixes move them on the full grid (about 1e-4 bohr); in the sectors
        # their averaged forces are zero and they stay.  The atoms with forces move as on the full grid.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            full = _run(root / "full", STEEPEST_DESCENT, "off")
            sectors = _run(root / "sectors", STEEPEST_DESCENT, "auto")
        for label, (status, _, log, errors) in (("full", full), ("sectors", sectors)):
            self.assertIn(status, (0, 4), f"{label}: {errors}\n{log[-3000:]}")
        moved, reference = np.asarray(sectors[1]["relaxation_positions"]), np.asarray(full[1]["relaxation_positions"])
        self.assertEqual(moved.shape, (2, 8, 3))
        np.testing.assert_allclose(moved[1, :4], moved[0, :4], rtol=0, atol=1e-12)
        self.assertGreater(float(np.abs(moved[1, 4:] - moved[0, 4:]).max()), 1.0e-3)
        np.testing.assert_allclose(moved[1, 4:], reference[1, 4:], rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            sectors[1]["relaxation_energies"], full[1]["relaxation_energies"], rtol=0, atol=1e-6
        )

    def test_steepest_descent_in_abelian_sectors_keeps_the_atoms_the_group_fixes(self) -> None:
        # C2v about the origin (Abelian sectors, every level solved): the forces and the moved positions are averaged
        # over its four operations, so the round-off of the forces across the axis of the atom on it moves nothing
        # (before, the simple scheme moved it by about Min_Step and the moved atoms left the group of the sectors)
        positions = displaced_positions(origin_shift=0.05)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            full = _run(root / "full", STEEPEST_DESCENT, "off", positions=positions)
            sectors = _run(root / "sectors", STEEPEST_DESCENT, "auto", positions=positions)
        for label, (status, _, log, errors) in (("full", full), ("sectors", sectors)):
            self.assertIn(status, (0, 4), f"{label}: {errors}\n{log[-3000:]}")
        self.assertIn("an Abelian group of 4 operations", sectors[2])
        self.assertIn("symmetrizing forces according to symmetry operations", sectors[2])
        moved = np.asarray(sectors[1]["relaxation_positions"])
        self.assertEqual(moved.shape, (2, 8, 3))
        # the atom on the axis keeps x and y; the atoms with forces move as on the full grid
        np.testing.assert_allclose(moved[1, 0, :2], moved[0, 0, :2], rtol=0, atol=1e-12)
        self.assertGreater(float(np.abs(moved[1] - moved[0]).max()), 1.0e-3)
        # every component moves as on the full grid or stays where the full grid moved it by its round-off force
        # alone (about Min_Step, below 1e-3 bohr)
        reference = np.asarray(full[1]["relaxation_positions"])
        same = np.abs(moved[1] - reference[1]) <= 1.0e-6
        kept = np.abs(moved[1] - moved[0]) <= 1.0e-12
        self.assertTrue(bool(np.all(same | kept)), f"{moved[1] - reference[1]}")
        self.assertTrue(bool(np.all(np.abs(reference[1] - reference[0])[~same] < 1.0e-3)))
        np.testing.assert_allclose(sectors[1]["relaxation_energies"], full[1]["relaxation_energies"], rtol=0, atol=1e-6)

    def test_dynamics_keeps_the_full_grid(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            full = _run(root / "full", SEEDED_DYNAMICS, "off")
            auto = _run(root / "auto", SEEDED_DYNAMICS, "auto")
            for label, (status, _, log, errors) in (("full", full), ("auto", auto)):
                self.assertEqual(status, 0, f"{label}: {errors}\n{log[-3000:]}")
            self.assertIn("skipped: periodic cell", auto[2])
            self.assertNotIn("Forces of the periodic cell in the sectors", auto[2])
            # the same calculation, bit for bit
            for key in ("dynamics_positions", "dynamics_energies", "dynamics_next_positions"):
                np.testing.assert_array_equal(auto[1][key], full[1][key], err_msg=key)
            for file in ("md_nrg.dat", "atom.cor"):
                self.assertEqual(_numbers(root / "auto" / file)[0].tolist(), _numbers(root / "full" / file)[0].tolist())


if __name__ == "__main__":
    unittest.main()
