"""Structural relaxation against Fortran PARSEC, and its own bookkeeping.

``tests/data/relaxation_parsec_reference.json`` holds relaxations of a water
molecule by Fortran PARSEC: for every structure the total energy and the
forces of ``parsec.out`` (eight and six decimals), and the files PARSEC wrote
(``atom.cor``, ``relax_restart.dat``, ``iterate.dat``) byte for byte.  The
cases are BFGS with and without symmetry, bounded BFGS, fixed atoms,
steepest descent, manual movements and runs continued with
``Relax_Restart``.

The tests come in four groups.

* The text files of ``Relaxation.files`` against the files of PARSEC.
* ``domove``: every Fortran trajectory is replayed from the energies and
  forces PARSEC printed.  No SCF calculation is needed for that, and the
  agreement is limited by the six printed decimals of the forces.
* The ionic loop, its recorder, the helpers of the command-line drivers and
  the readable driver itself on an analytic energy surface in place of the
  SCF calculation; and the input keywords.
* Complete relaxations through the readable command-line driver.  The
  reference cases above (sphere of 7.5 bohr, spacing 0.4 bohr) take four
  to five minutes each on that path and run only with
  ``PARSEC_TEST_RELAXATION_FULL=1``.  Four cases on the grid of the force
  fixture ``water_lda_micro`` with ``Expansion_Order: 4`` always run; they
  take most of the minute this module needs.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import functools
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

import numpy as np

from parsec_python.Forces import compute_forces, symmetrize_forces
from parsec_python.Input import (
    ANGSTROM_TO_BOHR,
    EV_TO_RYDBERG,
    ParsecInputError,
    UnsupportedParsecOptionError,
    parse_parsec_input,
)
from parsec_python.Relaxation import (
    IonicBackend,
    IonicObserver,
    LBFGSBState,
    MovementState,
    RelaxRestart,
    RelaxationRecorder,
    domove,
    format_atom_cor_energy,
    format_atom_cor_final,
    format_atom_cor_structure,
    format_relax_restart,
    format_xyz_frame,
    parse_manual_structures,
    parse_relax_restart,
    parse_xyz_frames,
    reference_backend,
    run_relaxation,
    steepest_descent_displacement,
)
from parsec_python.Relaxation.backend import release_orbitals
from parsec_python.Relaxation.domove import ERROR_LBFGSB, ERROR_LBFGSB_ABNORMAL
from parsec_python.Relaxation.files import BOHR_TO_ANGSTROM, RYDBERG_TO_EV
from parsec_python.Relaxation.run import (
    manual_structures,
    relaxation_header_lines,
    restart_structure,
    run_recorded_relaxation,
    run_single_point_with_forces,
    symmetry_decision,
)
from parsec_python.SCF import run_scf, update_geometry, warm_start_for_moved_atoms
from parsec_python.V_ion import center_cluster_geometry
import parsec_python.cli as cli_module
from parsec_python.cli import main as cli_main, relaxation_exit_code
from parsec_python.models import Atom, RelaxationSettings
from parsec_python.tests.force_reference import write_case


DATA = Path(__file__).resolve().parents[1] / "data"
CASES = json.loads(
    (DATA / "relaxation_parsec_reference.json").read_text(encoding="utf-8")
)["cases"]

POTENTIALS = {"O": "water_O_POTRE.DAT", "H": "water_H_POTRE.DAT"}
SYMBOLS = ("O", "H", "H")

#: PARSEC prints coordinates with six decimals.  A trajectory replayed from
#: the printed forces (six decimals, of forces of a few tenths of a Ry/bohr)
#: follows the Fortran one within a few units of the last printed digit.
REPLAY_TOLERANCE = 5.0e-6
#: Near the minimum the line search works with differences of energies and
#: of forces, which amplifies the rounding of the printed numbers.
AMPLIFIED_REPLAY_TOLERANCE = 5.0e-5
#: Two runs agree in ``atom.cor`` when their numbers differ by at most one
#: unit of the last printed digit, plus rounding: five decimals for the
#: energy per atom [eV] and the largest force [Ry/bohr], six for coordinates.
ENERGY_PRINT = 2.0e-5
FORCE_PRINT = 2.0e-5
POSITION_PRINT = 2.0e-6

#: The water molecule of the references after PARSEC centred it: without
#: symmetry and with the four operations of C2v among those of the cube.
ASYMMETRIC = np.array(
    [[-0.05, -0.435, 0.03], [1.45, 0.135, -0.25], [-1.45, 0.435, 0.25]]
)
SYMMETRIC = np.array([[0.0, -0.2, 0.0], [1.6, 0.2, 0.0], [-1.6, 0.2, 0.0]])
ASYMMETRIC.setflags(write=False)
SYMMETRIC.setflags(write=False)

FULL_SIZE = os.environ.get("PARSEC_TEST_RELAXATION_FULL", "").strip() not in {"", "0"}


# ---------------------------------------------------------------------------
# Reading the fixture and the files of a run
# ---------------------------------------------------------------------------


def _array(name: str, index: int, key: str) -> np.ndarray:
    return np.array(CASES[name]["structures"][index][key], dtype=np.float64)


@functools.lru_cache(maxsize=None)
def _translation(name: str):
    """The parsed input of a fixture case."""

    with tempfile.TemporaryDirectory() as directory:
        return parse_parsec_input(
            write_case(CASES[name], Path(directory), parsec_boundary=False)
        )


def _first_atoms(translation) -> tuple[Atom, ...]:
    """The atoms of the first structure, as the drivers prepare them."""

    problem = translation.problem
    if problem.recenter_geometry:
        return center_cluster_geometry(problem.atoms)
    return tuple(problem.atoms)


_ENERGY_LINE = re.compile(
    r"^Total Energy = (.{12}) \[eV/at\], Max force =(.{9}) \[Ry/au\]$"
)
FINAL_HEADER = "  Final coordinates for next run "
MINIMUM_FOUND = " Local minimum of total energy was found "
MOVEMENTS_USED = " NOTICE: Maximum number of movements used."


def _read_atom_cor(text: str) -> SimpleNamespace:
    """Split an ``atom.cor`` file into its parts.

    ``steps`` holds one entry per ``Step #`` block: the label, the species
    labels of the atoms, the coordinates and, when the structure was
    computed, the two numbers of its energy line.  ``statements`` are the
    other non-blank lines in their order and ``final`` the closing block.
    """

    steps: list[dict] = []
    statements: list[str] = []
    final: list[list[float]] | None = None
    rows: list[list[float]] | None = None
    for line in text.splitlines():
        if line.startswith("Step #"):
            steps.append(
                {
                    "step": int(line[6:]),
                    "symbols": [],
                    "positions": [],
                    "energy": None,
                    "force": None,
                }
            )
            rows = steps[-1]["positions"]
            continue
        if line == FINAL_HEADER:
            final = rows = []
            continue
        match = _ENERGY_LINE.match(line)
        if match:
            steps[-1]["energy"] = float(match.group(1))
            steps[-1]["force"] = float(match.group(2))
            continue
        fields = line.split()
        try:
            numbers = [float(field) for field in fields]
        except ValueError:
            numbers = []
        if len(numbers) == 3:
            rows.append(numbers)
        elif len(fields) == 2 and fields[1].isdigit() and rows is not final:
            steps[-1]["symbols"] += [fields[0]] * int(fields[1])
        elif line.strip():
            statements.append(line)
    return SimpleNamespace(
        steps=steps,
        statements=statements,
        final=None if final is None else np.array(final),
    )


def _restart_blocks(text: str, atoms: int = 3) -> list[RelaxRestart]:
    """Every block of a ``relax_restart.dat`` that holds coordinates only.

    PARSEC closes the file only in a BFGS run: a steepest-descent or manual
    run appends one block of ``atoms + 3`` records per movement.
    """

    records = text.splitlines()
    size = atoms + 3
    return [
        parse_relax_restart("\n".join(records[start : start + size]) + "\n", atoms)
        for start in range(0, len(records), size)
    ]


# ---------------------------------------------------------------------------
# Stand-ins for the SCF calculation
# ---------------------------------------------------------------------------


def _molecule(positions, symbols=SYMBOLS) -> SimpleNamespace:
    """Stands in for a prepared system: the loop looks only at its atoms."""

    return SimpleNamespace(
        atoms=tuple(
            Atom(symbol, position) for symbol, position in zip(symbols, positions)
        )
    )


def _positions_of(system) -> np.ndarray:
    return np.array([atom.position for atom in system.atoms], dtype=np.float64)


class _Springs:
    """An analytic energy surface in place of the SCF calculation.

    Three harmonic springs join the atoms of a water-like molecule,

    ``E = E0 + k/2 [(|R_O - R_H1| - r0)^2 + (|R_O - R_H2| - r0)^2]
             + k'/2 (|R_H1 - R_H2| - d0)^2``

    in Ry and bohr.  The forces are exact, sum to zero and have the symmetry
    of the structure.  The arithmetic is written out with Python floats, so
    that every platform computes the same doubles.
    """

    PAIRS = ((0, 1, 0.6, 1.8), (0, 2, 0.6, 1.8), (1, 2, 0.15, 2.9))
    OFFSET = -34.0

    @classmethod
    def evaluate(cls, positions) -> tuple[float, np.ndarray]:
        rows = [[float(value) for value in row] for row in np.asarray(positions)]
        energy = cls.OFFSET
        forces = [[0.0, 0.0, 0.0] for _ in rows]
        for first, second, stiffness, length in cls.PAIRS:
            delta = [rows[first][axis] - rows[second][axis] for axis in range(3)]
            distance = math.sqrt(
                delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]
            )
            stretch = distance - length
            energy += 0.5 * stiffness * stretch * stretch
            for axis in range(3):
                pull = stiffness * stretch * delta[axis] / distance
                forces[first][axis] -= pull
                forces[second][axis] += pull
        return energy, np.array(forces)


#: A perturbation of the forces without any symmetry, like the noise an SCF
#: loop leaves in them.
_NOISE = np.array([[1.0, -2.0, 3.0], [0.5, 1.5, -1.0], [-2.5, 0.5, 2.0]])


class _SurfaceBackend:
    """The four operations of an execution path on :class:`_Springs`.

    Keeps what the loop asked for, so that a test can read the protocol.
    ``noise`` scales :data:`_NOISE` added to the forces;
    ``obeys_operations=False`` leaves them unsymmetrized; ``energies`` and
    ``force_values`` replace the energy or one force component of the SCF
    call with that index; ``unconverged`` names the calls whose SCF loop
    reports no convergence.
    """

    def __init__(
        self,
        *,
        noise: float = 0.0,
        obeys_operations: bool = True,
        energies: dict[int, float] | None = None,
        force_values: dict[int, float] | None = None,
        unconverged: tuple[int, ...] = (),
        scf_iterations: int = 2,
    ) -> None:
        self.noise = noise
        self.obeys_operations = obeys_operations
        self.energies = dict(energies or {})
        self.force_values = dict(force_values or {})
        self.unconverged = set(unconverged)
        self.scf_iterations = scf_iterations
        self.scf_calls: list[tuple[np.ndarray, object]] = []
        self.force_calls: list[tuple[bool, object]] = []
        self.geometry_calls: list[tuple[object, tuple[Atom, ...]]] = []
        self.warm_calls: list[tuple[object, object, object]] = []
        self.released: list[int] = []

    def run_scf(self, system, callback=None, warm_start=None):
        index = len(self.scf_calls)
        positions = _positions_of(system)
        self.scf_calls.append((positions, warm_start))
        if callback is not None:
            for number in range(1, self.scf_iterations + 1):
                callback(SimpleNamespace(iteration=number))
        energy, _ = _Springs.evaluate(positions)
        backend = self

        class Orbitals:
            def release_intermediate_storage(self) -> None:
                backend.released.append(index)

        return SimpleNamespace(
            index=index,
            energies=SimpleNamespace(total=self.energies.get(index, energy)),
            converged=index not in self.unconverged,
            iterations=self.scf_iterations,
            wavefunctions=Orbitals(),
        )

    def compute_forces(self, system, result, *, symmetrize=True, operations=None):
        self.force_calls.append((symmetrize, operations))
        _, forces = _Springs.evaluate(_positions_of(system))
        forces = forces + self.noise * _NOISE
        if operations is not None and self.obeys_operations:
            forces = symmetrize_forces(forces, operations)
        if result.index in self.force_values:
            forces[1, 2] = self.force_values[result.index]
        return SimpleNamespace(total=forces)

    def update_geometry(self, system, atoms):
        self.geometry_calls.append((system, tuple(atoms)))
        return SimpleNamespace(atoms=tuple(atoms))

    def warm_start(self, previous_system, previous_result, system):
        self.warm_calls.append((previous_system, previous_result, system))
        return ("continue from", previous_result.index)

    def backend(self, potential_extrapolation: str = "none") -> IonicBackend:
        # What the driver read for Potential_Extrapolation.
        self.extrapolation = potential_extrapolation
        return IonicBackend(
            run_scf=self.run_scf,
            compute_forces=self.compute_forces,
            update_geometry=self.update_geometry,
            warm_start=self.warm_start,
        )


class _FortranReplay:
    """The energies and forces PARSEC printed, in place of the SCF calculation.

    The k-th structure the loop asks for gets the numbers of the k-th
    structure of the Fortran run, whatever its coordinates: those differ
    from the Fortran ones only by what the rounding of the printed forces
    does to the optimizer.
    """

    def __init__(self, structures: list[dict]) -> None:
        self.structures = structures
        self.positions: list[np.ndarray] = []

    def run_scf(self, system, callback=None, warm_start=None):
        index = len(self.positions)
        if index >= len(self.structures):
            raise AssertionError("the loop asks for more structures than PARSEC computed")
        self.positions.append(_positions_of(system))
        return SimpleNamespace(
            index=index,
            energies=SimpleNamespace(total=self.structures[index]["energy"]),
            converged=True,
            iterations=1,
        )

    def compute_forces(self, system, result, *, symmetrize=True, operations=None):
        return SimpleNamespace(
            total=np.array(self.structures[result.index]["forces"], dtype=np.float64)
        )

    def backend(self) -> IonicBackend:
        return IonicBackend(
            run_scf=self.run_scf,
            compute_forces=self.compute_forces,
            update_geometry=lambda system, atoms: SimpleNamespace(atoms=tuple(atoms)),
            warm_start=lambda previous_system, previous_result, system: None,
        )


class _Reporter:
    """Stands in for the text reporter of a run."""

    def __init__(self, write=None, translation=None) -> None:
        self.iterations: list[object] = []
        self.finished: list[tuple[object, object]] = []

    def header(self) -> None:
        """What the driver prints before the set-up."""

    def setup(self, system) -> None:
        """What the driver prints about the prepared system."""

    def iteration(self, iteration) -> None:
        self.iterations.append(iteration)

    def finish(self, result, seconds, forces=None) -> None:
        self.finished.append((result, forces))


class _Events(IonicObserver):
    """Writes down the events of an ionic loop."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self.movements: list[tuple[int, object, object, int]] = []
        self.outcome = None

    def scf_start(self, step, system) -> None:
        self.names.append(f"scf_start {step}")

    def scf_iteration(self, step, iteration) -> None:
        self.names.append(f"scf_iteration {step}.{iteration.iteration}")

    def scf_finish(self, step, system, result, seconds) -> None:
        self.names.append(f"scf_finish {step}")

    def forces(self, step, system, forces) -> None:
        self.names.append(f"forces {step}")

    def movement(self, step, record, report, state) -> None:
        self.names.append(f"movement {step}")
        self.movements.append((step, record, report, state.number))

    def finish(self, outcome) -> None:
        self.names.append("finish")
        self.outcome = outcome


class _ScratchCase(unittest.TestCase):
    """A test with its own directory, holding the water pseudopotentials."""

    def setUp(self) -> None:
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.directory = Path(scratch.name)

    def write_input(self, lines, *, parsec_boundary: bool = False) -> Path:
        return write_case(
            {"input": list(lines), "pseudopotentials": POTENTIALS},
            self.directory,
            parsec_boundary=parsec_boundary,
        )

    def parse(self, lines):
        return parse_parsec_input(self.write_input(lines))

    def write_file(self, name: str, text: str) -> Path:
        path = self.directory / name
        path.write_text(text, encoding="utf-8", newline="\n")
        return path


#: Lines that put the molecule into a periodic cubic cell of 9 bohr.
PERIODIC_DOMAIN = (
    "Periodic_System: .true.",
    "Boundary_Conditions: bulk",
    "begin Cell_Shape",
    "  9.0  9.0  9.0",
    "end Cell_Shape",
)


def _water_lines(
    control: str = "",
    *,
    oxygen_flag: str | None = None,
    hydrogen_flag: str | None = None,
    oxygen=("0.0  0.0  0.0",),
    hydrogen=("1.6  0.4  0.0", "-1.6  0.4  0.0"),
    domain=("Boundary_Sphere_Radius: 5.5",),
) -> list[str]:
    """A water input; the symmetric molecule unless coordinates are given."""

    lines = [*domain, "Grid_Spacing: 0.55", "States_Num: 8", "Atom_Types_Num: 2"]
    lines += [line for line in control.splitlines() if line.strip()]
    lines += ["Atom_Type: O", "Local_Component: p"]
    if oxygen_flag is not None:
        lines.append(f"Move_Flag: {oxygen_flag}")
    lines += ["begin Atom_Coord", *oxygen, "end Atom_Coord"]
    lines += ["Atom_Type: H", "Local_Component: s"]
    if hydrogen_flag is not None:
        lines.append(f"Move_Flag: {hydrogen_flag}")
    lines += ["begin Atom_Coord", *hydrogen, "end Atom_Coord"]
    return lines


# ---------------------------------------------------------------------------
# files.py
# ---------------------------------------------------------------------------


class AtomCorFileTests(unittest.TestCase):
    def test_formatters_reproduce_every_fortran_file(self) -> None:
        """The three formatters rebuild each ``atom.cor`` of PARSEC byte for byte.

        The numbers are read from the Fortran text and written again, which
        pins the layout: ``Step #`` label, species line, ``3(3x,f10.6)``
        coordinates, the energy line with its blank record, the closing
        statements and the ``3(2x,f11.6)`` final block.
        """

        for name, case in CASES.items():
            with self.subTest(case=name):
                text = case["files"]["atom.cor"]
                parts = _read_atom_cor(text)
                self.assertEqual(
                    [block["step"] for block in parts.steps],
                    list(range(len(case["structures"]))),
                )
                rebuilt = ""
                for block in parts.steps:
                    self.assertEqual(block["symbols"], list(SYMBOLS))
                    rebuilt += format_atom_cor_structure(
                        block["step"], block["symbols"], block["positions"]
                    )
                    rebuilt += format_atom_cor_energy(block["energy"], block["force"])
                rebuilt += "".join(f"{statement}\n" for statement in parts.statements)
                rebuilt += format_atom_cor_final(parts.final)
                self.assertEqual(rebuilt, text)

    def test_species_lines_follow_the_atom_types(self) -> None:
        """One line per run of equal labels with its atom count, as PARSEC's types."""

        positions = np.arange(15, dtype=float).reshape(5, 3) / 7.0
        text = format_atom_cor_structure(123, ["Si", "Si", "H", "H", "H"], positions)
        lines = text.splitlines()
        self.assertEqual(lines[0], "Step # 123")
        self.assertEqual(lines[1], " Si               2")
        self.assertEqual(lines[4], " H                3")
        self.assertEqual(len(lines), 1 + 2 + 5)
        self.assertEqual(lines[2], "     0.000000     0.142857     0.285714")
        self.assertTrue(text.endswith("\n"))

    def test_field_widths(self) -> None:
        """``3(3x,f10.6)`` for a structure and ``3(2x,f11.6)`` for the final block.

        The two layouts differ only for a coordinate that fills its field.
        """

        positions = np.array([[-12.3456789, 0.0, 1.5], [-123.456789, 99.5, -0.25]])
        self.assertEqual(
            format_atom_cor_structure(0, ["H"], positions[:1]),
            "Step #   0\n H                1\n   -12.345679     0.000000     1.500000\n",
        )
        self.assertEqual(
            format_atom_cor_final(positions),
            "\n  Final coordinates for next run \n"
            "   -12.345679     0.000000     1.500000\n"
            "  -123.456789    99.500000    -0.250000\n",
        )
        self.assertEqual(
            format_atom_cor_energy(-1234.567891, 12.345678),
            "Total Energy =  -1234.56789 [eV/at], Max force = 12.34568 [Ry/au]\n\n",
        )

    def test_energy_line_carries_the_energy_per_atom_of_parsec(self) -> None:
        """``atom.cor`` holds E/N in eV with PARSEC's 13.6058 eV per Ry.

        Checked on every structure of every Fortran run: the total energy of
        ``parsec.out`` gives the energy per atom of ``atom.cor`` within its
        five decimals.  With the CODATA value the numbers would differ by
        1e-3 eV.
        """

        self.assertEqual(RYDBERG_TO_EV, 13.6058)
        for name, case in CASES.items():
            for structure in case["structures"]:
                self.assertAlmostEqual(
                    RYDBERG_TO_EV * structure["energy"] / len(SYMBOLS),
                    structure["energy_per_atom_ev"],
                    delta=5.1e-6,
                    msg=name,
                )


class RelaxRestartFileTests(unittest.TestCase):
    BFGS_FILES = (
        ("water_bfgs_part1", "relax_restart.dat"),
        ("water_bfgs_restricted", "relax_restart.dat"),
        ("water_bfgs_fixed", "relax_restart.dat"),
        ("water_coarse_bfgs_part1", "relax_restart.dat"),
        ("water_coarse_bfgs_symmetric_fixed", "relax_restart.dat"),
        ("water_bfgs_part2_from_python", "relax_restart_in.dat"),
    )

    def test_fortran_bfgs_file_is_read(self) -> None:
        """Coordinates, counters and the optimizer state of a PARSEC file.

        The run was stopped after two movements, inside a line search.  The
        coordinates are those PARSEC announces in ``atom.cor`` for the next
        run.
        """

        case = CASES["water_bfgs_part1"]
        restart = parse_relax_restart(case["files"]["relax_restart.dat"], 3)
        self.assertEqual(restart.positions.shape, (3, 3))
        self.assertEqual(restart.positions[0, 0], -0.16324068870562949)
        self.assertEqual(restart.positions[2, 1], 0.56052915537872694)
        np.testing.assert_allclose(
            restart.positions,
            _read_atom_cor(case["files"]["atom.cor"]).final,
            rtol=0,
            atol=5.1e-7,
        )
        self.assertEqual((restart.iteration, restart.movable_atoms), (2, 3))
        self.assertEqual(restart.bound_step, -1.0)
        np.testing.assert_array_equal(restart.lower, np.zeros(9))
        np.testing.assert_array_equal(restart.upper, np.zeros(9))
        state = restart.lbfgsb
        self.assertEqual((state.n, state.m), (9, 7))
        self.assertEqual(state.task, "FG_LNSRCH")
        self.assertEqual(state.csave, "FG")
        # iterate.dat of this run: one iteration finished after three
        # evaluations; the fourth is the structure the file asks for.
        self.assertEqual(state.iteration, 1)
        self.assertEqual(state.function_evaluations, 4)

    def test_bfgs_files_are_written_back_byte_for_byte(self) -> None:
        """Parsing a BFGS restart file and writing it again changes no byte.

        Five files are PARSEC's (unrestricted, bounded, one and two movable
        atoms, coarse grid) and one was written by this package and then
        read by PARSEC, which continued the relaxation from it.
        """

        for name, file_name in self.BFGS_FILES:
            with self.subTest(case=name):
                text = CASES[name]["files"][file_name]
                restart = parse_relax_restart(text, 3)
                self.assertIsNotNone(restart.lbfgsb)
                self.assertEqual(format_relax_restart(restart), text)

    def test_bounds_of_a_restricted_run_surround_the_first_structure(self) -> None:
        """``Max_Step`` 0.08: every coordinate is confined to x0 -/+ 0.08.

        The bounds are those of the first structure of the relaxation, 30
        movements later, and PARSEC writes them as lower(j), upper(j) pairs.
        """

        restart = parse_relax_restart(
            CASES["water_bfgs_restricted"]["files"]["relax_restart.dat"], 3
        )
        start = _array("water_bfgs_restricted", 0, "positions").reshape(-1)
        self.assertEqual(restart.bound_step, 0.08)
        np.testing.assert_allclose(restart.lower, start - 0.08, rtol=0, atol=1e-15)
        np.testing.assert_allclose(restart.upper, start + 0.08, rtol=0, atol=1e-15)
        coordinates = restart.positions.reshape(-1)
        self.assertTrue(np.all(coordinates >= restart.lower))
        self.assertTrue(np.all(coordinates <= restart.upper))

    def test_file_of_a_run_with_fixed_atoms(self) -> None:
        """All atoms are listed; the optimizer state is that of the movable ones."""

        restart = parse_relax_restart(
            CASES["water_bfgs_fixed"]["files"]["relax_restart.dat"], 3
        )
        self.assertEqual((restart.iteration, restart.movable_atoms), (30, 1))
        self.assertEqual(restart.lbfgsb.n, 3)
        self.assertEqual(restart.lbfgsb.iwa.shape, (9,))
        start = _array("water_bfgs_fixed", 0, "positions")
        np.testing.assert_allclose(
            restart.positions[[0, 2]], start[[0, 2]], rtol=0, atol=1e-14
        )
        self.assertGreater(np.max(np.abs(restart.positions[1] - start[1])), 0.1)

    def test_files_without_optimizer_state(self) -> None:
        """Steepest descent and manual runs save coordinates and counters only.

        PARSEC closes the file only in a BFGS run, so these runs append a
        block per movement; like ``initial.f90``, the parser reads the first.
        """

        for name, blocks in (("water_simple", 7), ("water_manual", 3)):
            with self.subTest(case=name):
                text = CASES[name]["files"]["relax_restart.dat"]
                restart = parse_relax_restart(text, 3)
                self.assertIsNone(restart.lbfgsb)
                self.assertIsNone(restart.lower)
                self.assertIsNone(restart.bound_step)
                self.assertEqual((restart.iteration, restart.movable_atoms), (0, 3))
                self.assertEqual(
                    format_relax_restart(restart), "".join(text.splitlines(True)[:6])
                )
                appended = _restart_blocks(text)
                self.assertEqual(
                    [block.iteration for block in appended], list(range(blocks))
                )
                # Block k holds the structure computed next, number k + 1.
                for block in appended[: blocks - 1]:
                    np.testing.assert_allclose(
                        block.positions,
                        _array(name, block.iteration + 1, "positions"),
                        rtol=0,
                        atol=5.1e-7,
                    )

    def test_written_numbers_are_read_back_exactly(self) -> None:
        """Seventeen significant digits: text and doubles survive a round trip."""

        positions = np.array(
            [
                [1.0 / 3.0, -2.0 / 3.0, 1.0e-12],
                [123456.789, -0.1, 0.0],
                [5.0e-5, 7.25, -1.0e15],
            ]
        )
        plain = RelaxRestart(positions=positions, iteration=41, movable_atoms=2)
        text = format_relax_restart(plain)
        again = parse_relax_restart(text, 3)
        np.testing.assert_array_equal(again.positions, positions)
        self.assertEqual((again.iteration, again.movable_atoms), (41, 2))
        self.assertIsNone(again.lbfgsb)
        self.assertEqual(format_relax_restart(again), text)

        # An optimizer state with numbers in it: two BFGS moves on the springs.
        settings = RelaxationSettings(method="bfgs", maximum_step=0.3)
        state = MovementState()
        current = ASYMMETRIC
        for _ in range(2):
            energy, forces = _Springs.evaluate(current)
            current = domove(
                state, settings, current, [True] * 3, forces, energy
            ).positions
            state.number += 1
        full = RelaxRestart(
            positions=current,
            iteration=1,
            movable_atoms=3,
            lbfgsb=state.lbfgsb,
            lower=state.lower,
            upper=state.upper,
            bound_step=state.bound_step,
        )
        text = format_relax_restart(full)
        again = parse_relax_restart(text, 3)
        np.testing.assert_array_equal(again.positions, current)
        np.testing.assert_array_equal(again.lower, ASYMMETRIC.reshape(-1) - 0.3)
        np.testing.assert_array_equal(again.upper, ASYMMETRIC.reshape(-1) + 0.3)
        self.assertEqual(again.bound_step, 0.3)
        for field in ("wa", "dsave", "isave", "iwa", "lsave"):
            np.testing.assert_array_equal(
                getattr(again.lbfgsb, field), getattr(state.lbfgsb, field), err_msg=field
            )
        self.assertEqual(again.lbfgsb.task, state.lbfgsb.task)
        self.assertEqual(again.lbfgsb.csave, state.lbfgsb.csave)
        self.assertTrue(np.any(again.lbfgsb.wa != 0.0))
        self.assertEqual(format_relax_restart(again), text)

    def test_file_of_this_package_that_parsec_continued_from(self) -> None:
        """Same records as PARSEC's own file at the same point of the same run.

        ``relax_restart_in.dat`` was written by this package after two
        movements of ``water_bfgs_part1`` and read by Fortran PARSEC, which
        then followed the trajectory of its own continuation.  The two files
        have the same number of records and the same optimizer bookkeeping;
        the reals differ by what two SCF loops converged to 1e-8 Ry leave.
        """

        ours_text = CASES["water_bfgs_part2_from_python"]["files"]["relax_restart_in.dat"]
        theirs_text = CASES["water_bfgs_part1"]["files"]["relax_restart.dat"]
        self.assertEqual(len(ours_text.splitlines()), len(theirs_text.splitlines()))
        ours = parse_relax_restart(ours_text, 3)
        theirs = parse_relax_restart(theirs_text, 3)
        np.testing.assert_allclose(ours.positions, theirs.positions, rtol=0, atol=1e-7)
        self.assertEqual((ours.iteration, ours.movable_atoms), (2, 3))
        self.assertEqual(ours.bound_step, theirs.bound_step)
        self.assertEqual(ours.lbfgsb.task, theirs.lbfgsb.task)
        self.assertEqual(ours.lbfgsb.csave, theirs.lbfgsb.csave)
        np.testing.assert_array_equal(ours.lbfgsb.lsave, theirs.lbfgsb.lsave)
        self.assertEqual(ours.lbfgsb.iteration, theirs.lbfgsb.iteration)
        self.assertEqual(
            ours.lbfgsb.function_evaluations, theirs.lbfgsb.function_evaluations
        )
        self.assertAlmostEqual(
            ours.lbfgsb.projected_gradient_norm,
            theirs.lbfgsb.projected_gradient_norm,
            delta=1e-6,
        )
        # The first structures of the two Fortran continuations agree.
        np.testing.assert_allclose(
            _array("water_bfgs_part2_from_python", 1, "positions"),
            _array("water_bfgs_part2", 1, "positions"),
            rtol=0,
            atol=POSITION_PRINT,
        )

    def test_damaged_files_are_refused(self) -> None:
        """A file cut short or with records that lack numbers is named as such."""

        text = CASES["water_bfgs_part1"]["files"]["relax_restart.dat"]
        records = text.splitlines()
        with self.assertRaisesRegex(ValueError, "too short"):
            parse_relax_restart("\n".join(records[:5]) + "\n", 3)
        with self.assertRaisesRegex(
            ValueError, "three coordinates for each of the 3 atoms"
        ):
            parse_relax_restart(text.replace(records[2], "  1.5  0.25"), 3)
        with self.assertRaisesRegex(ValueError, "iteration and atom counters"):
            parse_relax_restart(text.replace(records[5], "  2"), 3)
        with self.assertRaisesRegex(ValueError, "records"):
            parse_relax_restart("\n".join(records[:200]) + "\n", 3)

    def test_file_for_another_number_of_atoms_is_refused(self) -> None:
        """The file holds no atom count; its layout must fit the structure.

        Read for more atoms, the header of the counters stands where
        coordinates are expected; read for fewer, coordinates stand where the
        header is, and the counters would be taken from an atom.
        """

        text = CASES["water_bfgs_part1"]["files"]["relax_restart.dat"]
        for atoms in (1, 2, 4, 5):
            with self.subTest(atoms=atoms):
                with self.assertRaisesRegex(
                    ValueError, f"three coordinates for each of the {atoms} atoms"
                ):
                    parse_relax_restart(text, atoms)


class ManualFileTests(unittest.TestCase):
    def test_fortran_manual_file(self) -> None:
        """``manual.dat`` of the reference run: the structures PARSEC computed next."""

        for name in ("water_manual", "water_coarse_manual"):
            with self.subTest(case=name):
                structures = parse_manual_structures(
                    CASES[name]["files"]["manual.dat"], 3, 2
                )
                self.assertEqual(len(structures), 2)
                for index, structure in enumerate(structures, start=1):
                    np.testing.assert_array_equal(
                        structure, _array(name, index, "positions")
                    )

    def test_records_are_read_like_a_list_directed_read(self) -> None:
        """Blank records, values over several records, commas, D exponents.

        A record that still holds values after the third coordinate loses
        them, and so does one cut by a comment sign.
        """

        text = (
            "\n"
            "  1.0  2.0\n"
            "  3.0  99.0\n"
            "\n"
            " 4.0D0, 5.0d-1, -6.0E0  7.0\n"
            " 0.5 0.25 0.125 ! third atom\n"
        )
        (structure,) = parse_manual_structures(text, 3, 1)
        np.testing.assert_array_equal(
            structure, [[1.0, 2.0, 3.0], [4.0, 0.5, -6.0], [0.5, 0.25, 0.125]]
        )

    def test_one_record_is_skipped_between_structures(self) -> None:
        """Exactly one, whatever it holds: text, nothing, or numbers."""

        rows = "1 2 3\n4 5 6\n"
        expected = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        for separator in ("second structure follows\n", "\n", "7 8 9\n"):
            with self.subTest(separator=separator):
                structures = parse_manual_structures(
                    rows + separator + rows + separator + rows, 2, 3
                )
                self.assertEqual(len(structures), 3)
                for structure in structures:
                    np.testing.assert_array_equal(structure, expected)

    def test_only_the_requested_structures_are_read(self) -> None:
        """``Movement_Num`` structures are read; what follows them is not looked at."""

        text = CASES["water_manual"]["files"]["manual.dat"]
        self.assertEqual(parse_manual_structures(text, 3, 0), [])
        (first,) = parse_manual_structures(text, 3, 1)
        np.testing.assert_array_equal(first, _array("water_manual", 1, "positions"))

    def test_file_that_ends_early_is_refused(self) -> None:
        """The message names the structure in which the file runs out."""

        text = CASES["water_manual"]["files"]["manual.dat"]
        with self.assertRaisesRegex(ValueError, "ends inside structure 3"):
            parse_manual_structures(text, 3, 3)
        with self.assertRaisesRegex(ValueError, "ends inside structure 1"):
            parse_manual_structures("1.0 2.0 3.0\n4.0 5.0\n", 2, 1)


class XyzTrajectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.positions = ASYMMETRIC
        self.energy, self.forces = _Springs.evaluate(ASYMMETRIC)

    def test_frames_are_read_back_in_bohr(self) -> None:
        """Positions are written in angstrom with ten decimals and read back."""

        moved = self.positions + 0.125
        text = format_xyz_frame(
            SYMBOLS, self.positions, step=0, energy=self.energy, forces=self.forces
        ) + format_xyz_frame(SYMBOLS, moved, step=1)
        frames = parse_xyz_frames(text)
        self.assertEqual(len(frames), 2)
        for (labels, positions), expected in zip(frames, (self.positions, moved)):
            self.assertEqual(labels, list(SYMBOLS))
            np.testing.assert_allclose(positions, expected, rtol=0, atol=1e-9)
        # A blank line between frames, as some programs write, is skipped.
        spaced = text.replace("\n3\nstep=1", "\n\n3\nstep=1")
        self.assertNotEqual(spaced, text)
        self.assertEqual(len(parse_xyz_frames(spaced)), 2)

    def test_units_and_comment_line(self) -> None:
        """Angstrom, eV and eV/angstrom with PARSEC's conversion constants."""

        self.assertEqual(BOHR_TO_ANGSTROM, 0.529177)
        text = format_xyz_frame(
            SYMBOLS, self.positions, step=7, energy=self.energy, forces=self.forces
        )
        lines = text.splitlines()
        self.assertEqual(lines[0], "3")
        fields = lines[1].split(" ", 4)
        self.assertEqual(fields[0], "step=7")
        self.assertAlmostEqual(
            float(fields[1].removeprefix("energy=")), 13.6058 * self.energy, delta=1e-9
        )
        self.assertAlmostEqual(
            float(fields[2].removeprefix("energy_Ry=")), self.energy, delta=1e-10
        )
        self.assertEqual(fields[3], "Properties=species:S:1:pos:R:3:forces:R:3")
        self.assertEqual(fields[4], 'pbc="F F F"')
        for line, symbol, position, force in zip(
            lines[2:], SYMBOLS, self.positions, self.forces
        ):
            columns = line.split()
            self.assertEqual(columns[0], symbol)
            np.testing.assert_allclose(
                [float(value) for value in columns[1:4]],
                position * 0.529177,
                rtol=0,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                [float(value) for value in columns[4:7]],
                force * 13.6058 / 0.529177,
                rtol=0,
                atol=1e-9,
            )

    def test_frame_without_energy_or_forces_and_with_element_names(self) -> None:
        """Positions alone: four columns; ``elements`` replaces species labels."""

        text = format_xyz_frame(
            ("O_core", "H", "H"), self.positions, step=0, elements=("O", "H", "H")
        )
        lines = text.splitlines()
        self.assertEqual(lines[1], 'step=0 Properties=species:S:1:pos:R:3 pbc="F F F"')
        self.assertEqual([line.split()[0] for line in lines[2:]], ["O", "H", "H"])
        self.assertTrue(all(len(line.split()) == 4 for line in lines[2:]))

    def test_damaged_files_are_refused(self) -> None:
        """No frame, a frame cut short, a bad atom count, a short atom line."""

        text = format_xyz_frame(SYMBOLS, self.positions, step=0)
        with self.assertRaisesRegex(ValueError, "holds no structure"):
            parse_xyz_frames("\n\n")
        with self.assertRaisesRegex(ValueError, "ends inside a frame"):
            parse_xyz_frames("".join(text.splitlines(True)[:-1]))
        with self.assertRaisesRegex(ValueError, "line 1 should hold the number of atoms"):
            parse_xyz_frames("three\n" + text)
        with self.assertRaisesRegex(ValueError, "too short"):
            parse_xyz_frames("1\ncomment\nH 0.0 0.0\n")


# ---------------------------------------------------------------------------
# domove.py
# ---------------------------------------------------------------------------


def _replay(name: str, *, structures: int | None = None, output_level: int | None = None):
    """Hand ``domove`` what PARSEC printed, structure after structure.

    The loop is that of ``parsec.F90``: the movement counter grows after
    every call that moved the atoms, and the run ends when the report says
    so or the movements are used up.  Returns the coordinates each structure
    was given, the reports and the state.
    """

    case = CASES[name]
    translation = _translation(name)
    settings = translation.relaxation
    restart = None
    if "restart_from" in case:
        restart = parse_relax_restart(
            CASES[case["restart_from"]]["files"]["relax_restart.dat"], 3
        )
    elif "relax_restart_in.dat" in case["files"]:
        restart = parse_relax_restart(case["files"]["relax_restart_in.dat"], 3)
    manual = None
    if settings.method == "manual":
        manual = parse_manual_structures(
            case["files"]["manual.dat"], 3, settings.max_movements
        )
    state = MovementState()
    current = _array(name, 0, "positions")
    used, reports = [], []
    for index, structure in enumerate(case["structures"][:structures]):
        used.append(current)
        report = domove(
            state,
            settings,
            current,
            translation.movable,
            np.array(structure["forces"]),
            structure["energy"],
            manual_structures=manual,
            restart=restart if index == 0 else None,
            output_level=(
                translation.output_level if output_level is None else output_level
            ),
        )
        reports.append(report)
        if report.minimum_force_found or report.done:
            break
        current = report.positions
        state.number += 1
        if state.number > settings.max_movements:
            break
    return used, reports, state


def _largest_difference(used, name: str) -> list[float]:
    return [
        float(np.max(np.abs(positions - _array(name, index, "positions"))))
        for index, positions in enumerate(used)
    ]


class SteepestDescentTests(unittest.TestCase):
    SETTINGS = RelaxationSettings(
        method="simple", minimum_force=0.005, maximum_step=0.06, minimum_step=0.0005
    )

    def test_displacement_saturates_between_the_two_steps(self) -> None:
        """A component of size Force_Min moves by Min_Step, a huge one by Max_Step.

        The constant of the law is chosen such that a component 1000
        Force_Min above Force_Min moves by 50 Min_Step.
        """

        gradient = np.array([[0.005, -0.005, 1.0e6], [-1.0e6, 5.005, -5.005]])
        displacement = steepest_descent_displacement(gradient, self.SETTINGS)
        np.testing.assert_allclose(
            displacement,
            [[-0.0005, 0.0005, -0.06], [0.06, -50 * 0.0005, 50 * 0.0005]],
            rtol=1e-13,
            atol=0,
        )

    def test_atoms_move_along_the_force_component_by_component(self) -> None:
        """Each Cartesian component moves on its own, by more the larger it is."""

        magnitudes = np.array([1.0e-4, 0.005, 0.01, 0.1, 0.3, 1.0, 10.0, 100.0])
        steps = -steepest_descent_displacement(magnitudes, self.SETTINGS)
        self.assertTrue(np.all(np.diff(steps) > 0.0))
        self.assertTrue(np.all((steps > 0.0) & (steps < 0.06)))
        np.testing.assert_array_equal(
            steepest_descent_displacement(-magnitudes, self.SETTINGS), steps
        )
        mixed = np.array([[0.3, -1.0, 0.01]])
        np.testing.assert_array_equal(
            steepest_descent_displacement(mixed, self.SETTINGS),
            [[-steps[4], steps[5], -steps[2]]],
        )

    def test_only_an_exactly_vanishing_component_stays(self) -> None:
        """The law does not go to zero with the force: almost Min_Step remains."""

        displacement = steepest_descent_displacement(
            np.array([0.0, -0.0, 1.0e-12, -1.0e-12]), self.SETTINGS
        )
        np.testing.assert_array_equal(displacement[:2], [0.0, 0.0])
        self.assertAlmostEqual(displacement[2], -0.000468, delta=1e-6)
        self.assertEqual(displacement[3], -displacement[2])

    def test_fortran_steps_obey_the_law(self) -> None:
        """Six movements of PARSEC, from the 17 digits of its restart file.

        Block k of ``relax_restart.dat`` holds structure k + 1 in full
        precision, so every displacement is known to 1e-16 and the law is
        checked through the printed forces alone: their rounding (5e-7)
        changes a displacement by less than 1e-8 bohr.
        """

        blocks = _restart_blocks(CASES["water_simple"]["files"]["relax_restart.dat"])
        settings = _translation("water_simple").relaxation
        self.assertEqual(settings, replace(self.SETTINGS, max_movements=6))
        before = _array("water_simple", 0, "positions")
        for index, block in enumerate(blocks[:6]):
            forces = _array("water_simple", index, "forces")
            np.testing.assert_allclose(
                block.positions - before,
                steepest_descent_displacement(-forces, settings),
                rtol=0,
                atol=1e-8,
                err_msg=f"movement {index}",
            )
            before = block.positions

    def test_fortran_trajectory_is_replayed(self) -> None:
        """``domove`` follows all seven structures; the movements are used up."""

        used, reports, state = _replay("water_simple")
        self.assertEqual(len(used), 7)
        self.assertLess(max(_largest_difference(used, "water_simple")), 1.5e-6)
        self.assertTrue(all(report.moved for report in reports))
        self.assertEqual(state.number, 7)
        self.assertIsNone(state.lbfgsb)
        # The last block PARSEC appended to its restart file.
        last = _restart_blocks(CASES["water_simple"]["files"]["relax_restart.dat"])[-1]
        np.testing.assert_allclose(
            reports[-1].positions, last.positions, rtol=0, atol=1e-7
        )


class BfgsReplayTests(unittest.TestCase):
    #: case: structures PARSEC computed, and how many of the first ones the
    #: replay follows within the print precision.
    RUNS = {
        "water_bfgs": (8, 8),
        "water_bfgs_asym": (17, 7),
        "water_bfgs_part1": (3, 3),
        "water_bfgs_part2": (14, 4),
        "water_bfgs_part2_from_python": (14, 4),
        "water_bfgs_fixed": (31, 7),
        "water_coarse_bfgs_part1": (3, 3),
        "water_coarse_bfgs_part2": (2, 2),
        "water_coarse_bfgs_symmetric_fixed": (2, 2),
    }

    def test_first_move_is_a_unit_step_along_the_forces(self) -> None:
        """Without second-derivative information L-BFGS-B steps by 1 bohr.

        The first structure after the input is ``x0 + F/|F|`` over the
        coordinates of the movable atoms, and it is the second structure of
        the Fortran run.
        """

        for name in (
            "water_bfgs",
            "water_bfgs_asym",
            "water_bfgs_fixed",
            "water_coarse_bfgs_part1",
            "water_coarse_bfgs_symmetric_fixed",
        ):
            with self.subTest(case=name):
                translation = _translation(name)
                mask = np.array(translation.movable)
                start = _array(name, 0, "positions")
                forces = _array(name, 0, "forces")
                state = MovementState()
                report = domove(
                    state,
                    translation.relaxation,
                    start,
                    mask,
                    forces,
                    CASES[name]["structures"][0]["energy"],
                )
                self.assertTrue(report.moved)
                self.assertEqual(report.task, "FG_LNSRCH")
                expected = start.copy()
                expected[mask] += forces[mask] / np.linalg.norm(forces[mask])
                np.testing.assert_allclose(report.positions, expected, rtol=0, atol=1e-14)
                np.testing.assert_allclose(
                    report.positions,
                    _array(name, 1, "positions"),
                    rtol=0,
                    atol=REPLAY_TOLERANCE,
                )
                # An unrestricted run carries zero bounds that do not act.
                self.assertEqual(state.bound_step, -1.0)
                np.testing.assert_array_equal(state.lower, np.zeros(3 * mask.sum()))
                np.testing.assert_array_equal(state.upper, np.zeros(3 * mask.sum()))

    def test_first_restricted_move_goes_to_the_corner_of_the_box(self) -> None:
        """Bounded BFGS: forces above 0.08 Ry/bohr take a coordinate to its bound.

        Coordinates without a force stay, and the Fortran run did the same.
        """

        name = "water_bfgs_restricted"
        settings = _translation(name).relaxation
        self.assertEqual(settings.bfgs_bound, 0.08)
        start = _array(name, 0, "positions")
        forces = _array(name, 0, "forces")
        self.assertTrue(np.all((forces == 0.0) | (np.abs(forces) > 0.08)))
        energy = CASES[name]["structures"][0]["energy"]
        state = MovementState()
        report = domove(state, settings, start, [True] * 3, forces, energy)
        np.testing.assert_allclose(
            report.positions, start + 0.08 * np.sign(forces), rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            report.positions, _array(name, 1, "positions"), rtol=0, atol=5.1e-7
        )
        np.testing.assert_array_equal(state.lower, start.reshape(-1) - 0.08)
        np.testing.assert_array_equal(state.upper, start.reshape(-1) + 0.08)

    def test_fortran_trajectories_are_replayed(self) -> None:
        """Every structure of every BFGS run, from the printed energies and forces.

        The replay asks for the same number of structures as PARSEC computed
        and ends for the same reason.  Its coordinates agree with the
        printed ones within 5e-6 bohr while the steps are large, and within
        5e-5 bohr near the minimum, where the rounding of the printed forces
        enters the line search.  The largest force is that of ``atom.cor``.
        """

        for name, (count, close) in self.RUNS.items():
            with self.subTest(case=name):
                case = CASES[name]
                settings = _translation(name).relaxation
                used, reports, state = _replay(name)
                self.assertEqual(len(case["structures"]), count)
                self.assertEqual(len(used), count)
                differences = _largest_difference(used, name)
                self.assertLess(max(differences[:close]), REPLAY_TOLERANCE)
                self.assertLess(max(differences), AMPLIFIED_REPLAY_TOLERANCE)
                for report, structure in zip(reports, case["structures"]):
                    self.assertAlmostEqual(
                        report.maximum_force, structure["maximum_force"], delta=6.5e-6
                    )
                statements = _read_atom_cor(case["files"]["atom.cor"]).statements
                if statements == [MINIMUM_FOUND]:
                    self.assertTrue(reports[-1].minimum_force_found)
                    self.assertFalse(reports[-1].moved)
                    self.assertLess(reports[-1].maximum_force, settings.minimum_force)
                else:
                    self.assertEqual(statements, [MOVEMENTS_USED])
                    self.assertTrue(reports[-1].moved)
                    self.assertEqual(state.number, settings.max_movements + 1)
                self.assertTrue(all(report.moved for report in reports[:-1]))
                self.assertFalse(any(report.done or report.error for report in reports))

    def test_replay_writes_the_iterate_file_of_parsec(self) -> None:
        """Unit 8 of the optimizer is ``iterate.dat``, byte for byte.

        Four significant digits of the projected gradient and of the energy
        per iteration: the rounding of the printed forces does not reach
        them in these runs.
        """

        for name in (
            "water_bfgs", "water_bfgs_asym", "water_bfgs_part1", "water_coarse_bfgs_part1"
        ):
            with self.subTest(case=name):
                _, reports, _ = _replay(name)
                written = "".join(
                    text
                    for report in reports
                    for unit, text in report.optimizer_output
                    if unit == 8
                )
                self.assertEqual(written, CASES[name]["files"]["iterate.dat"])
                self.assertIn(" N =            9     M =            7", written)

    def test_restarted_run_continues_from_the_file(self) -> None:
        """One call with the state of ``relax_restart.dat`` gives PARSEC's next structure.

        The file of part 1 ends inside a line search.  With the energy and
        the forces of the restart structure the optimizer finishes that line
        search instead of starting with a unit step, for PARSEC's own file
        and for the file this package wrote.
        """

        for name in (
            "water_bfgs_part2", "water_bfgs_part2_from_python", "water_coarse_bfgs_part2"
        ):
            with self.subTest(case=name):
                used, reports, state = _replay(name, structures=1)
                self.assertEqual(state.lbfgsb.function_evaluations, 5)
                step = reports[0].positions - used[0]
                self.assertLess(np.linalg.norm(step), 0.5)
                np.testing.assert_allclose(
                    reports[0].positions,
                    _array(name, 1, "positions"),
                    rtol=0,
                    atol=REPLAY_TOLERANCE,
                )
                # The optimizer of the file is not touched by the run.
                source = CASES[name].get("restart_from", name)
                file_name = (
                    "relax_restart.dat" if source != name else "relax_restart_in.dat"
                )
                again = parse_relax_restart(CASES[source]["files"][file_name], 3)
                self.assertEqual(again.lbfgsb.function_evaluations, 4)

    def test_restricted_coordinates_stay_in_their_box(self) -> None:
        """The box is x0 -/+ Max_Step of the first structure during the whole run.

        The first 18 structures of the Fortran run are replayed; after
        them the optimizer works at the bounds on energy differences at
        round-off level, which a replay from printed numbers cannot follow.
        """

        name = "water_bfgs_restricted"
        start = _array(name, 0, "positions")
        used, reports, state = _replay(name, structures=18)
        self.assertEqual(len(used), 18)
        self.assertLess(max(_largest_difference(used, name)), REPLAY_TOLERANCE)
        for positions in used:
            self.assertLessEqual(np.max(np.abs(positions - start)), 0.08 + 1e-15)
        np.testing.assert_array_equal(state.lower, start.reshape(-1) - 0.08)
        np.testing.assert_array_equal(state.upper, start.reshape(-1) + 0.08)
        # The hydrogen atoms end at their bound in x, in PARSEC as here.
        np.testing.assert_allclose(used[-1][[1, 2], 0], [1.68, -1.68], rtol=0, atol=1e-15)
        for structure in CASES[name]["structures"]:
            self.assertLessEqual(
                np.max(np.abs(np.array(structure["positions"]) - start)), 0.08 + 5.1e-7
            )

    def test_fixed_atoms_do_not_move(self) -> None:
        """``Move_Flag``: only the first hydrogen atom is a variable.

        The other two atoms keep their coordinates to the bit, the optimizer
        has three variables, and the largest force is that of the movable
        atom although the fixed oxygen atom feels a larger one.
        """

        name = "water_bfgs_fixed"
        self.assertEqual(_translation(name).movable, (False, True, False))
        used, reports, state = _replay(name)
        self.assertEqual(state.lbfgsb.n, 3)
        self.assertEqual(state.current.shape, (1, 3))
        for positions in used:
            np.testing.assert_array_equal(positions[[0, 2]], used[0][[0, 2]])
        self.assertGreater(np.max(np.abs(used[-1][1] - used[0][1])), 0.1)
        forces = _array(name, 0, "forces")
        self.assertAlmostEqual(
            reports[0].maximum_force, np.linalg.norm(forces[1]), delta=1e-15
        )
        self.assertGreater(np.linalg.norm(forces[0]), reports[0].maximum_force + 0.1)

    def test_output_level_changes_the_report_not_the_path(self) -> None:
        """``Output_Level`` 3 makes the optimizer print more to PARSEC's ``out.*``."""

        quiet_used, quiet_reports, _ = _replay("water_bfgs_part1", output_level=2)
        loud_used, loud_reports, _ = _replay("water_bfgs_part1", output_level=3)
        for quiet, loud in zip(quiet_used, loud_used):
            np.testing.assert_array_equal(quiet, loud)

        def report_length(reports) -> int:
            return sum(
                len(text)
                for report in reports
                for unit, text in report.optimizer_output
                if unit == 9
            )

        self.assertGreater(report_length(loud_reports), report_length(quiet_reports))


class MoveReportTests(unittest.TestCase):
    BFGS = RelaxationSettings(method="bfgs", minimum_force=0.005)
    #: A force that does not depend on the positions, in Ry/bohr.
    FORCE = np.array([[0.3, -0.2, 0.1], [-0.25, 0.4, 0.15], [0.2, 0.1, -0.35]])

    def test_force_criterion_is_met_before_anything_moves(self) -> None:
        """Largest force below Force_Min: the structure is the answer.

        ``domove.f90`` returns before it initializes the optimizer, with the
        statement it writes to ``parsec.out`` and to ``atom.cor``.
        """

        forces = np.full((3, 3), 0.0028)  # |F| = 0.00485 on every atom
        state = MovementState()
        report = domove(state, self.BFGS, ASYMMETRIC, [True] * 3, forces, -34.0)
        self.assertTrue(report.minimum_force_found)
        self.assertTrue(state.minimum_force_found)
        self.assertFalse(report.moved)
        self.assertFalse(report.done)
        self.assertEqual(report.error, 0)
        np.testing.assert_array_equal(report.positions, ASYMMETRIC)
        self.assertAlmostEqual(report.maximum_force, 0.0028 * math.sqrt(3.0), delta=1e-15)
        self.assertEqual(report.messages, (MINIMUM_FOUND,))
        self.assertEqual(report.atom_cor_messages, (MINIMUM_FOUND[1:],))
        self.assertEqual(report.optimizer_output, ())
        self.assertIsNone(state.lbfgsb)
        self.assertIsNone(state.current)
        # The comparison is strict, as in the Fortran: a force equal to
        # Force_Min is not yet below it.
        exact = np.zeros((3, 3))
        exact[1, 0] = 0.005
        self.assertFalse(
            domove(
                MovementState(), self.BFGS, ASYMMETRIC, [True] * 3, exact, -34.0
            ).minimum_force_found
        )

    def test_force_criterion_met_during_a_run(self) -> None:
        """The last structure of ``water_bfgs``: found, and the optimizer untouched."""

        used, reports, state = _replay("water_bfgs")
        last = reports[-1]
        self.assertTrue(last.minimum_force_found)
        self.assertFalse(last.moved or last.done)
        np.testing.assert_array_equal(last.positions, used[-1])
        self.assertEqual(last.messages, (MINIMUM_FOUND,))
        self.assertEqual(last.optimizer_output, ())
        self.assertEqual(state.number, 7)
        self.assertEqual(state.lbfgsb.task[:2], "FG")
        self.assertFalse(any(report.minimum_force_found for report in reports[:-1]))

    def test_optimizer_ends_when_the_forces_push_against_the_bounds(self) -> None:
        """Bounded BFGS in a field of constant force: converged at the corner.

        The first move takes every coordinate to its bound.  There the
        projected gradient vanishes although the forces are far above
        Force_Min, and L-BFGS-B declares convergence: the report says done,
        not that the minimum force was found, and leaves the atoms where
        they are.
        """

        settings = replace(self.BFGS, maximum_step=0.08)
        state = MovementState()
        energy = -float(np.sum(self.FORCE * ASYMMETRIC))
        first = domove(state, settings, ASYMMETRIC, [True] * 3, self.FORCE, energy)
        corner = ASYMMETRIC + 0.08 * np.sign(self.FORCE)
        np.testing.assert_allclose(first.positions, corner, rtol=0, atol=1e-15)
        state.number += 1
        report = domove(
            state,
            settings,
            first.positions,
            [True] * 3,
            self.FORCE,
            -float(np.sum(self.FORCE * first.positions)),
        )
        self.assertTrue(report.done)
        self.assertTrue(state.done)
        self.assertFalse(report.minimum_force_found)
        self.assertFalse(report.moved)
        self.assertEqual(report.error, 0)
        self.assertEqual(report.task, "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL")
        self.assertEqual(
            report.messages,
            ("LBFGS minimization converged ", report.task, "LBFGS is terminating"),
        )
        self.assertEqual(
            report.atom_cor_messages,
            ("Local minimum of total energy was found", "LBFGS is terminating"),
        )
        np.testing.assert_array_equal(report.positions, first.positions)
        self.assertGreater(report.maximum_force, 0.3)

    def test_energy_that_does_not_answer_ends_the_run_abnormally(self) -> None:
        """Forces that promise a descent the energy never shows: error 542.

        L-BFGS-B gives up a line search after 20 evaluations.  ``domove``
        then reports an abnormal exit and keeps the last structure, as the
        Fortran does, which returns before it updates the coordinates.
        """

        state = MovementState()
        current = ASYMMETRIC
        reports = []
        for _ in range(25):
            report = domove(state, self.BFGS, current, [True] * 3, self.FORCE, -34.0)
            reports.append(report)
            if not report.moved:
                break
            current = report.positions
            state.number += 1
        self.assertEqual(len(reports), 21)
        report = reports[-1]
        self.assertTrue(report.done)
        self.assertFalse(report.minimum_force_found)
        self.assertEqual(report.error, ERROR_LBFGSB_ABNORMAL)
        self.assertEqual(report.error, 542)
        self.assertEqual(report.task, "ABNORMAL_TERMINATION_IN_LNSRCH")
        self.assertEqual(
            report.messages,
            ("Abnormal exit from LBFGS routines : ", report.task, "LBFGS is terminating"),
        )
        self.assertEqual(report.atom_cor_messages, ("LBFGS is terminating",))
        np.testing.assert_array_equal(report.positions, current)

    def test_optimizer_error_is_reported(self) -> None:
        """An ERROR task of L-BFGS-B becomes error 541.

        Here a restart file whose bounds exclude each other, in a run that
        makes them act.
        """

        restart = RelaxRestart(
            positions=ASYMMETRIC,
            iteration=0,
            movable_atoms=3,
            lbfgsb=LBFGSBState.start(9, 7),
            lower=np.ones(9),
            upper=-np.ones(9),
            bound_step=0.08,
        )
        report = domove(
            MovementState(),
            replace(self.BFGS, maximum_step=0.08),
            ASYMMETRIC,
            [True] * 3,
            self.FORCE,
            -34.0,
            restart=restart,
        )
        self.assertTrue(report.done)
        self.assertFalse(report.moved)
        self.assertEqual(report.error, ERROR_LBFGSB)
        self.assertEqual(report.error, 541)
        self.assertEqual(report.task, "ERROR: NO FEASIBLE SOLUTION")
        self.assertEqual(
            report.messages,
            ("ERROR in LBFGS routines : ", report.task, "LBFGS is terminating"),
        )
        self.assertEqual(report.atom_cor_messages, ("LBFGS is terminating",))

    def test_manual_movements_follow_the_list(self) -> None:
        """Structure k + 1 comes from ``manual.dat``; forces play no part.

        After the last structure nothing is read and the atoms stay.  A
        structure without any force is no reason to stop: Force_Min is zero
        in a manual run.
        """

        settings = RelaxationSettings(method="manual", max_movements=2)
        self.assertEqual(settings.force_threshold, 0.0)
        listed = [ASYMMETRIC + 0.1, ASYMMETRIC - 0.2]
        no_force = np.zeros((3, 3))
        state = MovementState()
        current = ASYMMETRIC
        for index in range(3):
            report = domove(
                state,
                settings,
                current,
                [True] * 3,
                no_force,
                -34.0,
                manual_structures=listed,
            )
            self.assertTrue(report.moved)
            self.assertFalse(report.minimum_force_found or report.done)
            self.assertEqual(report.messages, ())
            expected = listed[index] if index < 2 else listed[1]
            np.testing.assert_array_equal(report.positions, expected)
            current = report.positions
            state.number += 1

    def test_manual_list_must_hold_every_structure(self) -> None:
        """A missing or misshapen structure fails the movement that needs it."""

        settings = RelaxationSettings(method="manual", max_movements=2)
        arguments = (settings, ASYMMETRIC, [True] * 3, self.FORCE, -34.0)
        with self.assertRaisesRegex(ValueError, "structure for movement 1"):
            domove(MovementState(), *arguments)
        with self.assertRaisesRegex(ValueError, "structure for movement 2"):
            domove(
                MovementState(number=1, current=ASYMMETRIC),
                *arguments,
                manual_structures=[ASYMMETRIC],
            )
        with self.assertRaisesRegex(ValueError, "wrong number of atoms"):
            domove(MovementState(), *arguments, manual_structures=[ASYMMETRIC[:2]])

    def test_arguments_that_do_not_fit_are_refused(self) -> None:
        """One force and one flag per atom; a restart file for the same atoms."""

        with self.assertRaisesRegex(ValueError, "do not match"):
            domove(
                MovementState(), self.BFGS, ASYMMETRIC, [True] * 3, self.FORCE[:2], -34.0
            )
        with self.assertRaisesRegex(ValueError, "do not match"):
            domove(MovementState(), self.BFGS, ASYMMETRIC, [True] * 2, self.FORCE, -34.0)
        # The restart file of the run with one movable atom, for three.
        restart = parse_relax_restart(
            CASES["water_bfgs_fixed"]["files"]["relax_restart.dat"], 3
        )
        with self.assertRaisesRegex(ValueError, "another number of movable atoms"):
            domove(
                MovementState(),
                self.BFGS,
                ASYMMETRIC,
                [True] * 3,
                self.FORCE,
                -34.0,
                restart=restart,
            )

    def test_continuation_cannot_put_bounds_on_an_unrestricted_run(self) -> None:
        """``Max_Step`` added to the input of a continuation is refused.

        The file of an unrestricted run holds zero bounds.  ``domove.f90``
        lets them act when the new input gives ``Max_Step``, which confines
        every coordinate to zero: all atoms go to the origin.  The reverse is
        harmless: the bounds of a restricted run are not used when the new
        input leaves ``Max_Step`` out.
        """

        unrestricted = parse_relax_restart(
            CASES["water_bfgs_part1"]["files"]["relax_restart.dat"], 3
        )
        self.assertEqual(unrestricted.bound_step, -1.0)
        arguments = (
            unrestricted.positions,
            [True] * 3,
            _array("water_bfgs_part2", 0, "forces"),
            CASES["water_bfgs_part2"]["structures"][0]["energy"],
        )
        with self.assertRaisesRegex(
            ValueError, "Max_Step cannot restrict its continuation"
        ):
            domove(
                MovementState(),
                replace(self.BFGS, maximum_step=0.08),
                *arguments,
                restart=unrestricted,
            )
        self.assertTrue(
            domove(MovementState(), self.BFGS, *arguments, restart=unrestricted).moved
        )

        bounded_settings = replace(self.BFGS, maximum_step=0.2)
        state = MovementState()
        current = ASYMMETRIC
        for _ in range(2):
            energy, forces = _Springs.evaluate(current)
            current = domove(
                state, bounded_settings, current, [True] * 3, forces, energy
            ).positions
            state.number += 1
        bounded = RelaxRestart(
            positions=current,
            iteration=1,
            movable_atoms=3,
            lbfgsb=state.lbfgsb,
            lower=state.lower,
            upper=state.upper,
            bound_step=state.bound_step,
        )
        energy, forces = _Springs.evaluate(current)
        free = MovementState()
        report = domove(
            free, self.BFGS, current, [True] * 3, forces, energy, restart=bounded
        )
        self.assertTrue(report.moved)
        self.assertEqual(free.bound_step, -1.0)
        again = MovementState()
        domove(
            again, bounded_settings, current, [True] * 3, forces, energy, restart=bounded
        )
        self.assertEqual(again.bound_step, 0.2)
        np.testing.assert_array_equal(again.lower, ASYMMETRIC.reshape(-1) - 0.2)

    def test_input_positions_are_not_changed(self) -> None:
        """The report holds new arrays, and a fixed atom keeps its row in them."""

        start = ASYMMETRIC.copy()
        forces = self.FORCE.copy()
        report = domove(
            MovementState(), self.BFGS, start, [True, False, True], forces, -34.0
        )
        np.testing.assert_array_equal(start, ASYMMETRIC)
        np.testing.assert_array_equal(forces, self.FORCE)
        np.testing.assert_array_equal(report.positions[1], ASYMMETRIC[1])
        self.assertTrue(np.all(report.positions[[0, 2]] != ASYMMETRIC[[0, 2]]))


# ---------------------------------------------------------------------------
# ionic_loop.py
# ---------------------------------------------------------------------------


def _asymmetry(positions: np.ndarray) -> float:
    """How far a structure is from the C2v arrangement of :data:`SYMMETRIC`."""

    return float(
        max(
            abs(positions[0, 0]),
            abs(positions[1, 0] + positions[2, 0]),
            abs(positions[1, 1] - positions[2, 1]),
            np.max(np.abs(positions[:, 2])),
        )
    )


class IonicLoopTests(unittest.TestCase):
    #: Converges on the springs within a dozen structures.
    BFGS = RelaxationSettings(method="bfgs", max_movements=60, minimum_force=0.005)
    #: Does not converge within the movements the tests allow.
    TIGHT = RelaxationSettings(method="bfgs", max_movements=60, minimum_force=1.0e-5)

    def run_loop(self, settings, start=ASYMMETRIC, *, surface=None, **keywords):
        surface = _SurfaceBackend() if surface is None else surface
        outcome = run_relaxation(
            _molecule(start), settings, surface.backend(), **keywords
        )
        return outcome, surface

    def test_movement_num_allows_one_structure_more(self) -> None:
        """``Movement_Num`` M: at most M + 1 structures are computed.

        The run then reports that the movements are used up, and the
        coordinates it hands on are those the mover asked for next: structure
        M + 1 of a longer run.
        """

        longer, _ = self.run_loop(replace(self.TIGHT, max_movements=7))
        self.assertEqual(len(longer.records), 8)
        for movements in (1, 2, 5):
            with self.subTest(movements=movements):
                outcome, surface = self.run_loop(
                    replace(self.TIGHT, max_movements=movements)
                )
                self.assertEqual(len(outcome.records), movements + 1)
                self.assertEqual(
                    [record.step for record in outcome.records],
                    list(range(movements + 1)),
                )
                self.assertEqual(len(surface.scf_calls), movements + 1)
                self.assertTrue(outcome.movements_exhausted)
                self.assertFalse(outcome.minimum_force_found)
                self.assertFalse(outcome.done)
                self.assertEqual(outcome.state.number, movements + 1)
                for mine, theirs in zip(outcome.records, longer.records):
                    np.testing.assert_array_equal(mine.positions, theirs.positions)
                np.testing.assert_array_equal(
                    outcome.positions, longer.records[movements + 1].positions
                )
                self.assertEqual(outcome.result.index, movements)
                np.testing.assert_array_equal(
                    _positions_of(outcome.system), outcome.records[-1].positions
                )

    def test_run_stops_when_the_largest_force_is_below_force_min(self) -> None:
        """The structure that meets ``Force_Min`` is the last one and the result."""

        outcome, _ = self.run_loop(self.BFGS)
        count = len(outcome.records)
        self.assertTrue(4 < count < 20)
        self.assertTrue(outcome.minimum_force_found)
        self.assertFalse(outcome.movements_exhausted)
        self.assertFalse(outcome.done)
        self.assertEqual(outcome.error, 0)
        self.assertLess(outcome.records[-1].maximum_force, 0.005)
        self.assertTrue(
            all(record.maximum_force >= 0.005 for record in outcome.records[:-1])
        )
        self.assertLess(outcome.records[-1].energy, outcome.records[0].energy - 0.01)
        np.testing.assert_array_equal(outcome.positions, outcome.records[-1].positions)
        self.assertEqual(outcome.state.number, count - 1)

        # Exactly as many movements as the run needs: still found, not used up.
        exact, _ = self.run_loop(replace(self.BFGS, max_movements=count - 1))
        self.assertEqual(len(exact.records), count)
        self.assertTrue(exact.minimum_force_found)
        self.assertFalse(exact.movements_exhausted)
        # One fewer: the last structure is not computed.
        short, _ = self.run_loop(replace(self.BFGS, max_movements=count - 2))
        self.assertEqual(len(short.records), count - 1)
        self.assertFalse(short.minimum_force_found)
        self.assertTrue(short.movements_exhausted)
        np.testing.assert_array_equal(short.positions, outcome.records[-1].positions)

    def test_movement_num_zero_computes_the_input_structure_only(self) -> None:
        """One SCF calculation with forces; the mover is not called."""

        events = _Events()
        outcome, surface = self.run_loop(
            replace(self.BFGS, max_movements=0), observer=events
        )
        self.assertEqual(len(outcome.records), 1)
        self.assertEqual(len(surface.scf_calls), 1)
        self.assertEqual(surface.geometry_calls, [])
        self.assertFalse(outcome.minimum_force_found)
        self.assertFalse(outcome.movements_exhausted)
        self.assertFalse(outcome.done)
        np.testing.assert_array_equal(outcome.positions, ASYMMETRIC)
        self.assertEqual(outcome.state.number, 0)
        self.assertIsNone(outcome.state.lbfgsb)
        (movement,) = events.movements
        self.assertIsNone(movement[2])
        self.assertEqual(events.names[-2:], ["movement 0", "finish"])

    def test_record_of_a_structure(self) -> None:
        """Energy in Ry, per atom in eV with PARSEC's constant, largest force."""

        outcome, surface = self.run_loop(replace(self.BFGS, max_movements=1))
        energy, forces = _Springs.evaluate(ASYMMETRIC)
        record = outcome.records[0]
        self.assertEqual(record.step, 0)
        self.assertEqual(record.symbols, SYMBOLS)
        np.testing.assert_array_equal(record.positions, ASYMMETRIC)
        self.assertEqual(record.energy, energy)
        self.assertAlmostEqual(
            record.energy_per_atom_ev, 13.6058 * energy / 3.0, delta=1e-12
        )
        np.testing.assert_array_equal(record.forces.total, forces)
        self.assertAlmostEqual(
            record.maximum_force, np.max(np.linalg.norm(forces, axis=1)), delta=1e-15
        )
        self.assertTrue(record.converged)
        self.assertEqual(record.scf_iterations, surface.scf_iterations)
        self.assertGreaterEqual(record.seconds, 0.0)

    def test_fixed_atoms_stay_where_they_are(self) -> None:
        """Movable flags: the fixed atom keeps its coordinates to the bit.

        The largest force of a record counts the movable atoms only, and it
        is that force the run is stopped on.
        """

        outcome, _ = self.run_loop(self.BFGS, movable=[False, True, True])
        self.assertTrue(outcome.minimum_force_found)
        self.assertGreater(len(outcome.records), 3)
        for record in outcome.records:
            np.testing.assert_array_equal(record.positions[0], ASYMMETRIC[0])
            self.assertAlmostEqual(
                record.maximum_force,
                np.max(np.linalg.norm(record.forces.total[1:], axis=1)),
                delta=1e-15,
            )
        self.assertTrue(np.all(outcome.records[-1].positions[1:] != ASYMMETRIC[1:]))
        self.assertEqual(outcome.state.lbfgsb.n, 6)

        with self.assertRaisesRegex(ValueError, "at least one movable atom"):
            self.run_loop(self.BFGS, movable=[False, False, False])
        with self.assertRaisesRegex(ValueError, "one movable flag per atom"):
            self.run_loop(self.BFGS, movable=[True, True])

    def test_symmetric_structure_stays_symmetric(self) -> None:
        """The operations of the input structure are found once and kept.

        Every force evaluation is asked to symmetrize with the same four
        operations.  The positions the mover returns are averaged over them
        too: with forces that carry noise and are not symmetrized, every
        structure still has the symmetry to round-off, while the same run
        without symmetry loses it.
        """

        tight = replace(self.BFGS, minimum_force=5.0e-4)
        outcome, surface = self.run_loop(
            tight, SYMMETRIC, surface=_SurfaceBackend(noise=1.0e-6)
        )
        self.assertTrue(outcome.minimum_force_found)
        self.assertGreater(len(outcome.records), 5)
        first_operations = surface.force_calls[0][1]
        self.assertEqual(len(first_operations), 4)
        for symmetrize, operations in surface.force_calls:
            self.assertTrue(symmetrize)
            self.assertIs(operations, first_operations)
        self.assertLess(
            max(_asymmetry(record.positions) for record in outcome.records), 1e-13
        )

        noisy, _ = self.run_loop(
            tight,
            SYMMETRIC,
            surface=_SurfaceBackend(noise=1.0e-6, obeys_operations=False),
        )
        self.assertLess(
            max(_asymmetry(record.positions) for record in noisy.records), 1e-13
        )

        free, surface = self.run_loop(
            tight, SYMMETRIC, surface=_SurfaceBackend(noise=1.0e-6), symmetrize=False
        )
        self.assertEqual(surface.force_calls[0], (False, None))
        self.assertGreater(
            max(_asymmetry(record.positions) for record in free.records), 1e-6
        )

    def test_operations_that_exchange_fixed_and_movable_atoms_are_dropped(self) -> None:
        """One hydrogen atom fixed: the mirror that exchanges the two is not used."""

        _, surface = self.run_loop(
            replace(self.BFGS, max_movements=1), SYMMETRIC, movable=[True, True, False]
        )
        operations = surface.force_calls[0][1]
        self.assertEqual(len(operations), 2)
        for operation in operations:
            np.testing.assert_array_equal(operation.image, [0, 1, 2])
        # Oxygen fixed and both hydrogen atoms movable keeps all four.
        _, surface = self.run_loop(
            replace(self.BFGS, max_movements=1), SYMMETRIC, movable=[False, True, True]
        )
        self.assertEqual(len(surface.force_calls[0][1]), 4)

    def test_manual_run_computes_the_listed_structures(self) -> None:
        """Input structure, then the list; all atoms move, no symmetry, no stop."""

        settings = RelaxationSettings(method="manual", max_movements=2)
        # The second structure is the minimum of the springs: no force at all.
        height = math.sqrt(1.8 * 1.8 - 1.45 * 1.45)
        relaxed = np.array([[0.0, 0.0, 0.0], [1.45, height, 0.0], [-1.45, height, 0.0]])
        listed = [SYMMETRIC + 0.05, relaxed]
        outcome, surface = self.run_loop(
            settings, SYMMETRIC, manual_structures=listed, movable=[True, False, False]
        )
        self.assertEqual(len(outcome.records), 3)
        np.testing.assert_array_equal(outcome.records[0].positions, SYMMETRIC)
        np.testing.assert_array_equal(outcome.records[1].positions, listed[0])
        np.testing.assert_array_equal(outcome.records[2].positions, relaxed)
        self.assertLess(outcome.records[2].maximum_force, 1e-12)
        self.assertTrue(outcome.movements_exhausted)
        self.assertFalse(outcome.minimum_force_found)
        self.assertEqual(surface.force_calls, [(False, None)] * 3)
        np.testing.assert_array_equal(outcome.positions, relaxed)

        with self.assertRaisesRegex(ValueError, "structure for movement 2"):
            self.run_loop(settings, SYMMETRIC, manual_structures=listed[:1])

    def test_energy_or_force_that_is_not_finite_stops_the_run(self) -> None:
        """L-BFGS-B takes an infinite energy for a minimum: the loop refuses it."""

        for label, spoiled in (
            ("energy nan", dict(energies={1: float("nan")})),
            ("energy inf", dict(energies={1: float("inf")})),
            ("force nan", dict(force_values={1: float("nan")})),
            ("force inf", dict(force_values={1: float("-inf")})),
        ):
            with self.subTest(case=label):
                events = _Events()
                surface = _SurfaceBackend(**spoiled)
                with self.assertRaisesRegex(
                    FloatingPointError, "structure 1 is not finite"
                ):
                    self.run_loop(self.BFGS, surface=surface, observer=events)
                self.assertEqual(len(surface.scf_calls), 2)
                self.assertEqual(events.names[-1], "forces 1")
                self.assertNotIn("finish", events.names)
        with self.assertRaisesRegex(FloatingPointError, "structure 0 is not finite"):
            self.run_loop(self.BFGS, surface=_SurfaceBackend(energies={0: float("nan")}))

    def test_observer_sees_the_events_in_order(self) -> None:
        """SCF start, iterations, SCF end, forces, movement; ``finish`` once."""

        events = _Events()
        outcome, _ = self.run_loop(replace(self.TIGHT, max_movements=1), observer=events)
        self.assertEqual(
            events.names,
            [
                "scf_start 0",
                "scf_iteration 0.1",
                "scf_iteration 0.2",
                "scf_finish 0",
                "forces 0",
                "movement 0",
                "scf_start 1",
                "scf_iteration 1.1",
                "scf_iteration 1.2",
                "scf_finish 1",
                "forces 1",
                "movement 1",
                "finish",
            ],
        )
        self.assertIs(events.outcome, outcome)
        for step, record, report, number in events.movements:
            self.assertIs(record, outcome.records[step])
            self.assertTrue(report.moved)
            # The counter grows after the observer was told.
            self.assertEqual(number, step)
        np.testing.assert_array_equal(
            events.movements[0][2].positions, outcome.records[1].positions
        )

    def test_next_scf_loop_continues_from_the_previous_one(self) -> None:
        """Geometry update and warm start between two structures.

        The moved atoms keep their species and order; the SCF loop of
        structure k + 1 receives what the backend made of system and result
        of structure k; the orbitals of every result but the last are
        released.
        """

        outcome, surface = self.run_loop(replace(self.TIGHT, max_movements=2))
        self.assertEqual(len(surface.geometry_calls), 2)
        for (system, atoms), record in zip(surface.geometry_calls, outcome.records[1:]):
            self.assertEqual(tuple(atom.symbol for atom in atoms), SYMBOLS)
            np.testing.assert_array_equal(
                _positions_of(SimpleNamespace(atoms=atoms)), record.positions
            )
        self.assertEqual(
            [warm for _, warm in surface.scf_calls],
            [None, ("continue from", 0), ("continue from", 1)],
        )
        for index, (previous_system, previous_result, system) in enumerate(
            surface.warm_calls
        ):
            self.assertEqual(previous_result.index, index)
            np.testing.assert_array_equal(
                _positions_of(previous_system), outcome.records[index].positions
            )
            np.testing.assert_array_equal(
                _positions_of(system), outcome.records[index + 1].positions
            )
        self.assertEqual(surface.released, [0, 1])

        cold, surface = self.run_loop(
            replace(self.TIGHT, max_movements=2), warm_start=False
        )
        self.assertEqual([warm for _, warm in surface.scf_calls], [None, None, None])
        self.assertEqual(surface.warm_calls, [])
        for mine, theirs in zip(cold.records, outcome.records):
            np.testing.assert_array_equal(mine.positions, theirs.positions)

    def test_scf_loop_that_did_not_converge_is_remembered(self) -> None:
        """The run goes on, as PARSEC does, and says so at the end."""

        outcome, _ = self.run_loop(
            replace(self.TIGHT, max_movements=2),
            surface=_SurfaceBackend(unconverged=(1,)),
        )
        self.assertEqual(
            [record.converged for record in outcome.records], [True, False, True]
        )
        self.assertTrue(outcome.scf_failed)
        clean, _ = self.run_loop(replace(self.TIGHT, max_movements=2))
        self.assertFalse(clean.scf_failed)

    def test_optimizer_that_ends_stops_the_run(self) -> None:
        """A bounded run that converges at its bounds: done, not relaxed."""

        settings = replace(self.BFGS, maximum_step=0.02)
        outcome, _ = self.run_loop(settings)
        self.assertTrue(outcome.done)
        self.assertFalse(outcome.minimum_force_found)
        self.assertFalse(outcome.movements_exhausted)
        self.assertEqual(outcome.error, 0)
        self.assertTrue(outcome.state.lbfgsb.task.startswith("CONVERGENCE"))
        self.assertGreater(outcome.records[-1].maximum_force, 0.005)
        np.testing.assert_array_equal(outcome.positions, outcome.records[-1].positions)
        for record in outcome.records:
            self.assertLessEqual(
                np.max(np.abs(record.positions - ASYMMETRIC)), 0.02 + 1e-15
            )


class RestartedLoopTests(_ScratchCase):
    def continue_from_file(self, settings, movements, *, movable=(True, True, True)):
        """Stop a run after ``movements``, restart it from its file, join both."""

        recorder = RelaxationRecorder(
            reporter=_Reporter(),
            write=lambda text="": None,
            settings=settings,
            movable=movable,
            directory=self.directory,
        )
        first = run_relaxation(
            _molecule(ASYMMETRIC),
            replace(settings, max_movements=movements),
            _SurfaceBackend().backend(),
            movable=movable,
            observer=recorder,
        )
        restart = parse_relax_restart(
            (self.directory / "relax_restart.dat").read_text(encoding="utf-8"), 3
        )
        second = run_relaxation(
            _molecule(restart.positions),
            replace(
                settings,
                restart=True,
                max_movements=settings.max_movements - movements - 1,
            ),
            _SurfaceBackend().backend(),
            movable=movable,
            restart=restart,
        )
        return first, restart, second

    def test_restarted_run_continues_like_the_uninterrupted_one(self) -> None:
        """Stopped by ``Movement_Num`` and continued from ``relax_restart.dat``.

        The two parts compute, to the bit, the structures and energies of
        the run that was not stopped, and end as it does.  This holds for
        BFGS stopped inside a line search and after one, with a fixed atom
        and with bounds, whose box stays that of the first structure; and
        for steepest descent, which carries no state and here uses up its
        movements.
        """

        cases = (
            (
                "bfgs",
                RelaxationSettings(method="bfgs", max_movements=40, minimum_force=1e-3),
                (True,) * 3,
            ),
            (
                "fixed atom",
                RelaxationSettings(method="bfgs", max_movements=40, minimum_force=1e-3),
                (False, True, True),
            ),
            (
                "bounded",
                RelaxationSettings(
                    method="bfgs", max_movements=40, minimum_force=1e-3, maximum_step=0.2
                ),
                (True,) * 3,
            ),
            (
                "simple",
                RelaxationSettings(method="simple", max_movements=12, minimum_force=1e-3),
                (True,) * 3,
            ),
        )
        for label, settings, movable in cases:
            whole = run_relaxation(
                _molecule(ASYMMETRIC),
                settings,
                _SurfaceBackend().backend(),
                movable=movable,
            )
            self.assertGreater(len(whole.records), 6, label)
            for movements in (1, 2, 4):
                with self.subTest(case=label, movements=movements):
                    first, restart, second = self.continue_from_file(
                        settings, movements, movable=movable
                    )
                    self.assertEqual(len(first.records), movements + 1)
                    self.assertEqual(restart.iteration, movements)
                    self.assertEqual(restart.lbfgsb is None, settings.method == "simple")
                    joined = first.records + second.records
                    self.assertEqual(len(joined), len(whole.records))
                    for mine, theirs in zip(joined, whole.records):
                        np.testing.assert_array_equal(mine.positions, theirs.positions)
                        self.assertEqual(mine.energy, theirs.energy)
                    self.assertEqual(
                        second.minimum_force_found, whole.minimum_force_found
                    )
                    self.assertEqual(
                        second.movements_exhausted, whole.movements_exhausted
                    )
                    self.assertEqual(second.done, whole.done)
                    np.testing.assert_array_equal(second.positions, whole.positions)
                    if label == "bounded":
                        np.testing.assert_array_equal(
                            second.state.lower, ASYMMETRIC.reshape(-1) - 0.2
                        )
            self.assertEqual(
                whole.movements_exhausted, settings.method == "simple", label
            )

    def test_restart_file_is_used_for_the_first_structure_only(self) -> None:
        """A second movement does not go back to the state of the file."""

        settings = RelaxationSettings(method="bfgs", max_movements=40, minimum_force=1e-3)
        first, restart, second = self.continue_from_file(settings, 2)
        # Three structures were computed and the fourth was asked for.
        evaluations = restart.lbfgsb.function_evaluations
        self.assertEqual(evaluations, 4)
        self.assertEqual(
            second.state.lbfgsb.function_evaluations,
            evaluations + len(second.records) - 1,
        )
        # The parsed file is left as it was read.
        self.assertEqual(restart.lbfgsb.function_evaluations, evaluations)


# ---------------------------------------------------------------------------
# recorder.py and run.py
# ---------------------------------------------------------------------------


class RecorderTests(_ScratchCase):
    def record(
        self,
        settings,
        start=ASYMMETRIC,
        *,
        movable=(True,) * 3,
        directory="scratch",
        **keywords,
    ):
        reporter = _Reporter()
        log: list[str] = []
        recorder = RelaxationRecorder(
            reporter=reporter,
            write=lambda text="": log.append(text),
            settings=settings,
            movable=movable,
            directory=self.directory if directory == "scratch" else directory,
            elements=("O", "H", "H"),
        )
        outcome = run_relaxation(
            _molecule(start),
            settings,
            _SurfaceBackend().backend(),
            movable=movable,
            observer=recorder,
            **keywords,
        )
        return outcome, recorder, reporter, "\n".join(log)

    def read(self, name: str) -> str:
        return (self.directory / name).read_bytes().decode("ascii")

    def expected_atom_cor(self, outcome, statements) -> str:
        """Every computed structure with its energy line, statements, final block."""

        text = ""
        for record in outcome.records:
            text += format_atom_cor_structure(
                record.step, record.symbols, record.positions
            )
            text += format_atom_cor_energy(
                record.energy_per_atom_ev, record.maximum_force
            )
        text += "".join(f"{statement}\n" for statement in statements)
        return text + format_atom_cor_final(outcome.positions)

    def test_files_of_a_run_stopped_by_movement_num(self) -> None:
        """``atom.cor``, ``relax_restart.dat``, ``relaxation.xyz``, ``iterate.dat``.

        ``atom.cor`` lists the three structures that were computed, says
        that the movements are used up and closes with the structure the
        mover asked for next, which is also what the restart file holds with
        the optimizer state.
        """

        settings = RelaxationSettings(method="bfgs", max_movements=2, minimum_force=1e-5)
        outcome, recorder, reporter, log = self.record(settings)
        self.assertEqual(
            sorted(path.name for path in self.directory.iterdir()),
            ["atom.cor", "iterate.dat", "relax_restart.dat", "relaxation.xyz"],
        )
        self.assertEqual(
            list(dict.fromkeys(path.name for path in recorder.files)),
            ["atom.cor", "relaxation.xyz", "iterate.dat", "relax_restart.dat"],
        )
        self.assertNotIn(b"\r", (self.directory / "atom.cor").read_bytes())
        self.assertEqual(
            self.read("atom.cor"), self.expected_atom_cor(outcome, [MOVEMENTS_USED])
        )

        restart = parse_relax_restart(self.read("relax_restart.dat"), 3)
        np.testing.assert_array_equal(restart.positions, outcome.positions)
        self.assertEqual((restart.iteration, restart.movable_atoms), (2, 3))
        self.assertEqual(restart.bound_step, -1.0)
        self.assertEqual(restart.lbfgsb.task, outcome.state.lbfgsb.task)
        np.testing.assert_array_equal(restart.lbfgsb.wa, outcome.state.lbfgsb.wa)

        frames = parse_xyz_frames(self.read("relaxation.xyz"))
        self.assertEqual(len(frames), 3)
        for (labels, positions), record in zip(frames, outcome.records):
            self.assertEqual(labels, ["O", "H", "H"])
            np.testing.assert_allclose(positions, record.positions, rtol=0, atol=1e-9)
        comments = self.read("relaxation.xyz").splitlines()[1::5]
        self.assertEqual(
            [comment.split()[0] for comment in comments], ["step=0", "step=1", "step=2"]
        )
        self.assertTrue(all(":forces:R:3" in comment for comment in comments))

        iterate = self.read("iterate.dat")
        self.assertTrue(iterate.startswith("RUNNING THE L-BFGS-B CODE\n"))
        self.assertIn(" N =            9     M =            7", iterate)
        rows = [line for line in iterate.splitlines() if re.match(r"^ +\d+ +\d+ ", line)]
        self.assertEqual(len(rows), outcome.state.lbfgsb.iteration + 1)

        # The reporter of the run sees every SCF iteration and every result.
        self.assertEqual(len(reporter.iterations), 3 * 2)
        self.assertEqual([result.index for result, _ in reporter.finished], [0, 1, 2])
        for (_, forces), record in zip(reporter.finished, outcome.records):
            self.assertIs(forces, record.forces)
        for expected in (
            "Time for movement    0 is [sec]:",
            "  calling domove",
            " Structure of movement    2: the SCF loop continues from",
            " NOTICE: Maximum number of movements used.",
            "  WARNING: Minimum force is NOT satisfied!!",
            " Structures computed            :        3",
            " Files: atom.cor, relaxation.xyz, iterate.dat, relax_restart.dat",
        ):
            self.assertIn(expected, log)
        self.assertNotIn("SCF convergence not achieved", log)

    def test_files_of_a_run_that_meets_the_force_criterion(self) -> None:
        """``atom.cor`` ends with the statement and the structure that met it.

        The restart file holds that structure too: the last one the mover
        asked for, written before it was computed.
        """

        settings = RelaxationSettings(
            method="bfgs", max_movements=60, minimum_force=0.005
        )
        outcome, _, _, log = self.record(settings)
        self.assertTrue(outcome.minimum_force_found)
        self.assertEqual(
            self.read("atom.cor"), self.expected_atom_cor(outcome, [MINIMUM_FOUND])
        )
        parts = _read_atom_cor(self.read("atom.cor"))
        self.assertEqual(len(parts.steps), len(outcome.records))
        np.testing.assert_allclose(
            parts.final, outcome.records[-1].positions, rtol=0, atol=5.1e-7
        )
        restart = parse_relax_restart(self.read("relax_restart.dat"), 3)
        np.testing.assert_array_equal(restart.positions, outcome.records[-1].positions)
        self.assertEqual(restart.iteration, len(outcome.records) - 2)
        self.assertIn("  Minimum force is satisfied ", log)
        self.assertIn(MINIMUM_FOUND, log)
        self.assertNotIn("NOTICE", log)

    def test_files_of_a_single_structure(self) -> None:
        """``Movement_Num`` 0: the structure and its closing block, no energy line."""

        settings = RelaxationSettings(method="bfgs", max_movements=0)
        outcome, _, _, log = self.record(settings)
        self.assertEqual(
            self.read("atom.cor"),
            format_atom_cor_structure(0, SYMBOLS, ASYMMETRIC)
            + format_atom_cor_final(ASYMMETRIC),
        )
        self.assertEqual(
            sorted(path.name for path in self.directory.iterdir()),
            ["atom.cor", "relaxation.xyz"],
        )
        self.assertEqual(len(parse_xyz_frames(self.read("relaxation.xyz"))), 1)
        self.assertNotIn("calling domove", log)

    def test_restart_file_without_optimizer(self) -> None:
        """Steepest descent, manual movements and fixed atoms in the restart file.

        Runs without BFGS write coordinates and counters only.  Unlike
        PARSEC, which appends to the file in these runs, the file holds the
        last movement.  A manual run writes it once more after its last
        structure, with the same coordinates.
        """

        simple = RelaxationSettings(method="simple", max_movements=3)
        outcome, _, _, _ = self.record(simple, movable=(False, True, True))
        text = self.read("relax_restart.dat")
        self.assertEqual(len(text.splitlines()), 6)
        restart = parse_relax_restart(text, 3)
        self.assertIsNone(restart.lbfgsb)
        self.assertEqual((restart.iteration, restart.movable_atoms), (3, 2))
        np.testing.assert_array_equal(restart.positions, outcome.positions)
        self.assertFalse((self.directory / "iterate.dat").exists())

        manual = RelaxationSettings(method="manual", max_movements=2)
        listed = [ASYMMETRIC + 0.1, ASYMMETRIC - 0.1]
        outcome, _, _, log = self.record(manual, manual_structures=listed)
        restart = parse_relax_restart(self.read("relax_restart.dat"), 3)
        self.assertEqual((restart.iteration, restart.movable_atoms), (2, 3))
        np.testing.assert_array_equal(restart.positions, listed[1])
        self.assertEqual(
            self.read("atom.cor"), self.expected_atom_cor(outcome, [MOVEMENTS_USED])
        )
        self.assertEqual(len(_read_atom_cor(self.read("atom.cor")).steps), 3)
        # A manual run has no force criterion to report on.
        self.assertNotIn("Minimum force", log)

    def test_recorder_without_a_directory_writes_nothing(self) -> None:
        """``directory=None``: the run is reported and no file appears."""

        settings = RelaxationSettings(method="bfgs", max_movements=1)
        outcome, recorder, reporter, log = self.record(settings, directory=None)
        self.assertEqual(recorder.files, [])
        self.assertEqual(list(self.directory.iterdir()), [])
        self.assertEqual(len(reporter.finished), 2)
        self.assertNotIn("Files:", log)


class FortranFilesTests(_ScratchCase):
    """The files of every Fortran run, written by the recorder of this package.

    The loop of ``run_recorded_relaxation`` is given the energies and forces
    PARSEC printed, so no SCF calculation is made.  ``atom.cor`` then has
    the structures, the numbers and the closing statements of the Fortran
    file.
    """

    def replay(self, name: str, *, structures: int | None = None):
        """Run the case in ``self.directory/name`` as a driver would."""

        case = CASES[name]
        directory = self.directory / name
        directory.mkdir()
        translation = _translation(name)
        if structures is not None:
            translation = replace(
                translation,
                relaxation=replace(translation.relaxation, max_movements=structures - 1),
            )
        if "manual.dat" in case["files"]:
            (directory / "manual.dat").write_text(
                case["files"]["manual.dat"], newline="\n"
            )
        if "restart_from" in case:
            (directory / "relax_restart.dat").write_text(
                CASES[case["restart_from"]]["files"]["relax_restart.dat"], newline="\n"
            )
        elif "relax_restart_in.dat" in case["files"]:
            (directory / "relax_restart.dat").write_text(
                case["files"]["relax_restart_in.dat"], newline="\n"
            )
        translation, restart, notes = restart_structure(translation, directory)
        use_symmetry, _ = symmetry_decision(translation)
        outcome = run_recorded_relaxation(
            translation,
            SimpleNamespace(atoms=_first_atoms(translation)),
            _FortranReplay(case["structures"]).backend(),
            _Reporter(),
            lambda text="": None,
            directory=directory,
            restart=restart,
            symmetrize=use_symmetry,
        )
        written = _read_atom_cor((directory / "atom.cor").read_bytes().decode("ascii"))
        return outcome, written, restart, notes

    def assert_same_numbers(
        self, written, reference, tolerance, *, first_block=True
    ) -> None:
        for mine, theirs in zip(written.steps, reference.steps):
            self.assertEqual(mine["step"], theirs["step"])
            self.assertEqual(mine["symbols"], theirs["symbols"])
            self.assertAlmostEqual(mine["energy"], theirs["energy"], delta=1.01e-5)
            self.assertAlmostEqual(mine["force"], theirs["force"], delta=1.01e-5)
            if first_block or mine["step"] > 0:
                np.testing.assert_allclose(
                    mine["positions"], theirs["positions"], rtol=0, atol=tolerance
                )

    def test_complete_runs(self) -> None:
        """Runs from the input structure to their end, structure by structure.

        The first structure is the input one after PARSEC's recentring, and
        the uncentred one in a manual run.
        """

        runs = {
            "water_bfgs": REPLAY_TOLERANCE,
            "water_bfgs_asym": AMPLIFIED_REPLAY_TOLERANCE,
            "water_simple": REPLAY_TOLERANCE,
            "water_manual": 1e-12,
            "water_bfgs_part1": REPLAY_TOLERANCE,
            "water_bfgs_fixed": AMPLIFIED_REPLAY_TOLERANCE,
            "water_coarse_bfgs_part1": REPLAY_TOLERANCE,
            "water_coarse_manual": 1e-12,
            "water_coarse_bfgs_symmetric_fixed": REPLAY_TOLERANCE,
        }
        for name, tolerance in runs.items():
            with self.subTest(case=name):
                outcome, written, restart, notes = self.replay(name)
                reference = _read_atom_cor(CASES[name]["files"]["atom.cor"])
                self.assertIsNone(restart)
                self.assertEqual(notes, [])
                self.assertEqual(len(written.steps), len(reference.steps))
                self.assert_same_numbers(written, reference, tolerance)
                self.assertEqual(written.statements, reference.statements)
                np.testing.assert_allclose(
                    written.final, reference.final, rtol=0, atol=tolerance
                )
                # The rounding of the printed forces reaches the fourth digit
                # of one projected gradient in the run with the fixed atoms.
                if "iterate.dat" in CASES[name]["files"] and name != "water_bfgs_fixed":
                    iterate = (self.directory / name / "iterate.dat").read_bytes()
                    self.assertEqual(
                        iterate.decode("ascii"), CASES[name]["files"]["iterate.dat"]
                    )
                self.assertEqual(
                    outcome.minimum_force_found, reference.statements == [MINIMUM_FOUND]
                )
                self.assertEqual(
                    outcome.movements_exhausted, reference.statements == [MOVEMENTS_USED]
                )

    def test_restarted_runs(self) -> None:
        """``Relax_Restart``: energies, forces and structures of the continuation.

        One block differs, by intention: PARSEC opens ``atom.cor`` with the
        coordinates of the input file, although the first structure it
        computes is that of the restart file; here ``Step # 0`` is the
        structure that was computed.
        """

        runs = {
            "water_bfgs_part2": AMPLIFIED_REPLAY_TOLERANCE,
            "water_bfgs_part2_from_python": AMPLIFIED_REPLAY_TOLERANCE,
            "water_coarse_bfgs_part2": REPLAY_TOLERANCE,
        }
        for name, tolerance in runs.items():
            with self.subTest(case=name):
                outcome, written, restart, notes = self.replay(name)
                reference = _read_atom_cor(CASES[name]["files"]["atom.cor"])
                self.assertEqual(
                    notes,
                    [
                        "WARNING: atom coordinates in input were replaced by "
                        "coordinates in relax_restart.dat",
                        " RESTART LBFGS FROM PREVIOUS RUN",
                    ],
                )
                self.assertEqual(len(written.steps), len(reference.steps))
                self.assert_same_numbers(written, reference, tolerance, first_block=False)
                self.assertEqual(written.statements, reference.statements)
                np.testing.assert_allclose(
                    written.final, reference.final, rtol=0, atol=tolerance
                )
                # Step # 0: the restart structure here, the input in PARSEC.
                np.testing.assert_allclose(
                    written.steps[0]["positions"], restart.positions, rtol=0, atol=5.1e-7
                )
                np.testing.assert_allclose(
                    written.steps[0]["positions"],
                    _array(name, 0, "positions"),
                    rtol=0,
                    atol=5.1e-7,
                )
                np.testing.assert_array_equal(reference.steps[0]["positions"], ASYMMETRIC)
                self.assertGreater(
                    np.max(np.abs(restart.positions - ASYMMETRIC)), 0.01
                )

    def test_restricted_run_up_to_where_the_optimizer_stalls(self) -> None:
        """The first 18 structures of the bounded run."""

        name = "water_bfgs_restricted"
        outcome, written, _, _ = self.replay(name, structures=18)
        reference = _read_atom_cor(CASES[name]["files"]["atom.cor"])
        self.assertEqual(len(written.steps), 18)
        self.assert_same_numbers(written, reference, REPLAY_TOLERANCE)
        self.assertTrue(outcome.movements_exhausted)
        restart = parse_relax_restart(
            (self.directory / name / "relax_restart.dat").read_bytes().decode("ascii"), 3
        )
        self.assertEqual(restart.bound_step, 0.08)
        np.testing.assert_array_equal(restart.lower, SYMMETRIC.reshape(-1) - 0.08)


class DriverHelperTests(_ScratchCase):
    def test_minimization_header_is_that_of_parsec(self) -> None:
        """What ``usrinputfile.F90`` prints under "Minimization data:".

        Line for line for BFGS and manual runs, followed here by the number
        of movable atoms.  Two lines are known to differ: the remark PARSEC
        adds to the steepest-descent scheme is left out, and the bound of a
        restricted run is printed with six decimals.
        """

        for name, case in CASES.items():
            with self.subTest(case=name):
                lines = relaxation_header_lines(_translation(name))
                movable = sum(case["movable"]) if "manual" not in name else 3
                self.assertEqual(
                    lines[-2:], [f" Movable atoms = {movable:6d} of      3", ""]
                )
                expected = [
                    line
                    for line in case["minimization_header"]
                    if "make sure the construction of displacements" not in line
                ]
                expected = [
                    " Maximum displacement of atoms from initial position is 0.080000"
                    if line.startswith(" Maximum displacement of atoms")
                    else line
                    for line in expected
                ]
                self.assertEqual(lines[:-2], expected)
        self.assertEqual(
            relaxation_header_lines(self.parse(_water_lines())), [" No minimization!", ""]
        )
        atomic = replace(_translation("water_bfgs"), potential_extrapolation="atomic")
        self.assertEqual(
            relaxation_header_lines(atomic)[-2:],
            [" Potential extrapolation after a move: atomic", ""],
        )

    def test_relax_restart_reads_the_file_beside_the_input(self) -> None:
        """``Relax_Restart``: coordinates and optimizer of ``relax_restart.dat``.

        The coordinates replace those of the input and are not recentred
        again: the run that wrote them had centred the cluster.
        """

        text = CASES["water_bfgs_part1"]["files"]["relax_restart.dat"]
        self.write_file("relax_restart.dat", text)
        translation = _translation("water_bfgs_part2")
        self.assertTrue(translation.relaxation.restart)
        self.assertTrue(translation.problem.recenter_geometry)
        moved, restart, notes = restart_structure(translation, self.directory)
        expected = parse_relax_restart(text, 3)
        np.testing.assert_array_equal(
            _positions_of(moved.problem), expected.positions
        )
        self.assertEqual(tuple(atom.symbol for atom in moved.problem.atoms), SYMBOLS)
        self.assertFalse(moved.problem.recenter_geometry)
        np.testing.assert_array_equal(
            _positions_of(SimpleNamespace(atoms=_first_atoms(moved))), expected.positions
        )
        self.assertEqual(restart.lbfgsb.task, "FG_LNSRCH")
        self.assertEqual(
            notes,
            [
                "WARNING: atom coordinates in input were replaced by coordinates "
                "in relax_restart.dat",
                " RESTART LBFGS FROM PREVIOUS RUN",
            ],
        )
        # The translation that was passed is left alone.
        np.testing.assert_array_equal(
            _positions_of(translation.problem),
            _positions_of(_translation("water_bfgs_part1").problem),
        )

    def test_relax_restart_without_a_file_starts_from_scratch(self) -> None:
        """As ``initial.f90``: a warning, the input structure, a fresh optimizer."""

        translation = _translation("water_bfgs_part2")
        same, restart, notes = restart_structure(translation, self.directory)
        self.assertIs(same, translation)
        self.assertIsNone(restart)
        self.assertEqual(
            notes,
            [
                "WARNING! relax_restart.dat not found; "
                "start structural relaxation from scratch"
            ],
        )

    def test_file_beside_the_input_is_not_read_unasked(self) -> None:
        """Without ``Relax_Restart`` the file beside the input is left alone."""

        self.write_file(
            "relax_restart.dat", CASES["water_bfgs_part1"]["files"]["relax_restart.dat"]
        )
        translation = _translation("water_bfgs_part1")
        self.assertFalse(translation.relaxation.restart)
        self.assertEqual(
            restart_structure(translation, self.directory), (translation, None, [])
        )

    def test_restart_from_accepts_each_format(self) -> None:
        """``--restart-from``: a file of PARSEC, of this package, or XYZ.

        A ``relax_restart.dat`` continues the optimizer of a BFGS run under
        any file name, at its coordinates as they are.  An XYZ file gives the
        coordinates of its last frame, converted to bohr; they are recentred
        like those of the input, and the optimizer starts afresh.
        """

        translation = _translation("water_bfgs_part1")
        fortran = self.write_file(
            "from_parsec.dat", CASES["water_bfgs_part1"]["files"]["relax_restart.dat"]
        )
        python = self.write_file(
            "from_python.dat",
            CASES["water_bfgs_part2_from_python"]["files"]["relax_restart_in.dat"],
        )
        frames = format_xyz_frame(SYMBOLS, ASYMMETRIC, step=0) + format_xyz_frame(
            SYMBOLS, ASYMMETRIC + 0.25, step=1, energy=-34.0, forces=np.ones((3, 3))
        )
        trajectory = self.write_file("earlier.XYZ", frames)

        for source in (fortran, python):
            with self.subTest(source=source.name):
                moved, restart, notes = restart_structure(
                    translation, self.directory, source
                )
                expected = parse_relax_restart(source.read_text(encoding="utf-8"), 3)
                np.testing.assert_array_equal(
                    _positions_of(moved.problem), expected.positions
                )
                self.assertFalse(moved.problem.recenter_geometry)
                self.assertEqual(restart.lbfgsb.task, "FG_LNSRCH")
                self.assertEqual(
                    notes,
                    [
                        "WARNING: atom coordinates in input were replaced by "
                        f"coordinates in {source.name}",
                        " RESTART LBFGS FROM PREVIOUS RUN",
                    ],
                )

        moved, restart, notes = restart_structure(translation, self.directory, trajectory)
        np.testing.assert_allclose(
            _positions_of(moved.problem), ASYMMETRIC + 0.25, rtol=0, atol=1e-9
        )
        self.assertTrue(translation.problem.recenter_geometry)
        self.assertTrue(moved.problem.recenter_geometry)
        self.assertIsNone(restart)
        self.assertEqual(
            notes,
            [
                "WARNING: atom coordinates in input were replaced by coordinates "
                "in earlier.XYZ"
            ],
        )

    def test_restart_from_keeps_the_optimizer_for_bfgs_only(self) -> None:
        """Another scheme, or a single point, takes the coordinates alone."""

        source = self.write_file(
            "relax_restart.dat", CASES["water_bfgs_part1"]["files"]["relax_restart.dat"]
        )
        expected = parse_relax_restart(source.read_text(encoding="utf-8"), 3).positions
        moved, restart, notes = restart_structure(
            _translation("water_simple"), self.directory, source
        )
        np.testing.assert_array_equal(_positions_of(moved.problem), expected)
        self.assertIsNone(restart.lbfgsb)
        self.assertEqual(len(notes), 1)

        single_point = self.parse(_water_lines())
        self.assertIsNone(single_point.relaxation)
        moved, _, notes = restart_structure(single_point, self.directory, source)
        np.testing.assert_array_equal(_positions_of(moved.problem), expected)
        self.assertEqual(len(notes), 1)

    def test_restart_file_for_other_atoms_is_refused(self) -> None:
        """Another atom count, a file cut short and a missing file are errors."""

        translation = _translation("water_bfgs_part1")
        two_atoms = self.write_file(
            "two.xyz", format_xyz_frame(("O", "H"), ASYMMETRIC[:2], step=0)
        )
        swapped = self.write_file(
            "swapped.xyz",
            format_xyz_frame(("H", "O", "H"), ASYMMETRIC, step=0),
        )
        with self.assertRaisesRegex(
            ValueError,
            "swapped.xyz: atom 1 is H, the input has O there; an XYZ file must "
            "list the atoms in the order of the input",
        ):
            restart_structure(_translation("water_bfgs_part1"), self.directory, swapped)
        with self.assertRaisesRegex(ValueError, "two.xyz holds 2 atoms, the input 3"):
            restart_structure(translation, self.directory, two_atoms)
        short = self.write_file(
            "short.dat", "  Atomic coordinates follow (all atoms)\n 0.0 0.0 0.0\n"
        )
        with self.assertRaisesRegex(ValueError, "too short"):
            restart_structure(translation, self.directory, short)
        with self.assertRaises(OSError):
            restart_structure(translation, self.directory, self.directory / "missing.dat")

    def test_symmetry_is_used_when_the_movable_flags_respect_it(self) -> None:
        """Which relaxations run with the point group of the structure.

        A fixed atom that an operation would exchange with a movable one
        switches symmetry off, with a remark for the log; manual movements
        and ``Ignore_Symmetry`` use none.
        """

        decisions = {
            "all atoms movable": (_water_lines("Minimization: BFGS"), True, False),
            "oxygen fixed": (
                _water_lines(
                    "Minimization: BFGS", oxygen_flag="-1", oxygen=("0.0 0.0 0.0  0",)
                ),
                True,
                False,
            ),
            "one hydrogen fixed": (
                _water_lines("Minimization: BFGS", oxygen_flag="0", hydrogen_flag="1"),
                False,
                True,
            ),
            "one hydrogen fixed, steepest descent": (
                _water_lines(
                    "Minimization: simple",
                    oxygen_flag="0",
                    hydrogen_flag="-1",
                    hydrogen=("1.6 0.4 0.0 0", "-1.6 0.4 0.0 1"),
                ),
                False,
                True,
            ),
            "manual": (
                _water_lines("Minimization: manual\nMovement_Num: 0"), False, False
            ),
            "ignore symmetry": (
                _water_lines("Minimization: BFGS\nIgnore_Symmetry: .true."), False, False
            ),
            "single point": (_water_lines(), True, False),
        }
        for label, (lines, expected, remark) in decisions.items():
            with self.subTest(case=label):
                use, note = symmetry_decision(self.parse(lines))
                self.assertEqual(use, expected)
                if remark:
                    self.assertIn("Move_Flag fixes atoms that a symmetry operation", note)
                    self.assertIn("uses no symmetry", note)
                else:
                    self.assertIsNone(note)

    def test_symmetry_is_judged_on_the_structure_the_run_starts_from(self) -> None:
        """A structure without symmetry has nothing a fixed atom could break.

        The input is centred first, as the calculation centres it: the
        symmetric molecule is given 3 bohr off the origin here.
        """

        self.assertEqual(
            symmetry_decision(_translation("water_bfgs_fixed")), (True, None)
        )
        shifted = _water_lines(
            "Minimization: BFGS",
            oxygen_flag="0",
            hydrogen_flag="1",
            oxygen=("3.0 3.0 3.0",),
            hydrogen=("4.6 3.4 3.0", "1.4 3.4 3.0"),
        )
        use, note = symmetry_decision(self.parse(shifted))
        self.assertFalse(use)
        self.assertIsNotNone(note)

    def test_manual_structures_are_read_beside_the_input(self) -> None:
        """``manual.dat`` is required exactly when a manual run moves the atoms."""

        translation = _translation("water_manual")
        with self.assertRaisesRegex(
            FileNotFoundError, "Movement_Num 2 needs manual.dat beside the input"
        ):
            manual_structures(translation, self.directory)
        self.write_file("manual.dat", CASES["water_manual"]["files"]["manual.dat"])
        listed = manual_structures(translation, self.directory)
        self.assertEqual(len(listed), 2)
        np.testing.assert_array_equal(listed[1], _array("water_manual", 2, "positions"))
        # No file is needed where nothing is read from it.
        self.assertIsNone(manual_structures(_translation("water_bfgs"), self.directory))
        (self.directory / "manual.dat").unlink()
        no_movement = replace(
            translation, relaxation=replace(translation.relaxation, max_movements=0)
        )
        self.assertIsNone(manual_structures(no_movement, self.directory))

    def test_recorded_run_reads_manual_dat_and_names_the_elements(self) -> None:
        """``run_recorded_relaxation``: the pieces a driver hands over.

        ``manual.dat`` comes from the input directory, the files go to the
        output directory, and the XYZ file names the elements of the
        pseudopotentials.
        """

        inputs = self.directory / "inputs"
        outputs = self.directory / "outputs"
        inputs.mkdir()
        outputs.mkdir()
        (inputs / "manual.dat").write_text(CASES["water_manual"]["files"]["manual.dat"])
        # The oxygen species carries a label that is not an element symbol.
        translation = parse_parsec_input(
            write_case(
                {
                    "input": [
                        line.replace("Atom_Type: O", "Atom_Type: O_core")
                        for line in CASES["water_manual"]["input"]
                    ],
                    "pseudopotentials": {"O_core": POTENTIALS["O"], "H": POTENTIALS["H"]},
                },
                inputs,
                parsec_boundary=False,
            )
        )
        surface = _SurfaceBackend()
        reporter = _Reporter()
        outcome = run_recorded_relaxation(
            translation,
            SimpleNamespace(atoms=tuple(translation.problem.atoms)),
            surface.backend(),
            reporter,
            lambda text="": None,
            directory=outputs,
            input_directory=inputs,
            symmetrize=False,
        )
        self.assertFalse((inputs / "atom.cor").exists())
        self.assertEqual(len(outcome.records), 3)
        for index, record in enumerate(outcome.records):
            np.testing.assert_array_equal(
                record.positions, _array("water_manual", index, "positions")
            )
        self.assertEqual(
            sorted(path.name for path in outputs.iterdir()),
            ["atom.cor", "relax_restart.dat", "relaxation.xyz"],
        )
        labels, _ = parse_xyz_frames((outputs / "relaxation.xyz").read_text())[0]
        self.assertEqual(labels, ["O", "H", "H"])
        self.assertEqual(outcome.records[0].symbols, ("O_core", "H", "H"))
        self.assertIn(" O_core", (outputs / "atom.cor").read_text().splitlines()[1])
        self.assertEqual(len(reporter.finished), 3)

    def test_single_point_with_forces(self) -> None:
        """One SCF loop from scratch, its forces, and one report of both."""

        surface = _SurfaceBackend()
        reporter = _Reporter()
        result, forces = run_single_point_with_forces(
            _molecule(SYMMETRIC), surface.backend(), reporter, symmetrize=False
        )
        self.assertEqual([warm for _, warm in surface.scf_calls], [None])
        self.assertEqual(surface.force_calls, [(False, None)])
        self.assertEqual(len(reporter.iterations), surface.scf_iterations)
        self.assertEqual(len(reporter.finished), 1)
        self.assertIs(reporter.finished[0][0], result)
        self.assertIs(reporter.finished[0][1], forces)
        np.testing.assert_array_equal(forces.total, _Springs.evaluate(SYMMETRIC)[1])

    def test_readable_backend_names_the_readable_operations(self) -> None:
        """SCF loop, forces, geometry update and warm start of the NumPy path."""

        backend = reference_backend()
        self.assertIs(backend.run_scf, run_scf)
        self.assertIs(backend.compute_forces, compute_forces)
        self.assertIs(backend.update_geometry, update_geometry)
        self.assertIs(backend.warm_start.func, warm_start_for_moved_atoms)
        self.assertEqual(backend.warm_start.keywords, {"extrapolation": "none"})
        atomic = reference_backend("atomic")
        self.assertEqual(atomic.warm_start.keywords, {"extrapolation": "atomic"})
        # Each loop of second order follows its own trajectory.
        histories = [
            reference_backend("second_order").warm_start.keywords["history"]
            for _ in range(2)
        ]
        self.assertIsNot(histories[0], histories[1])
        with self.assertRaisesRegex(ValueError, "Potential_Extrapolation"):
            reference_backend("linear")
        # A result with host orbitals has nothing to release.
        release_orbitals(SimpleNamespace(wavefunctions=np.zeros(3)))
        release_orbitals(SimpleNamespace())


# ---------------------------------------------------------------------------
# Input keywords
# ---------------------------------------------------------------------------


class RelaxationInputTests(_ScratchCase):
    def test_minimization_selects_the_scheme(self) -> None:
        """``Minimization``: bfgs, simple or manual; none is a single point."""

        for label, method in (
            ("BFGS", "bfgs"),
            ("bfgs", "bfgs"),
            ("Simple", "simple"),
            ("manual", "manual"),
        ):
            with self.subTest(label=label):
                translation = self.parse(_water_lines(f"Minimization: {label}"))
                self.assertEqual(translation.relaxation.method, method)
                self.assertTrue(translation.compute_forces)
        for label in ("none", "None"):
            with self.subTest(label=label):
                translation = self.parse(_water_lines(f"Minimization: {label}"))
                self.assertIsNone(translation.relaxation)
                self.assertFalse(translation.compute_forces)
        with self.assertRaisesRegex(
            ParsecInputError, "unknown minimization method: 'cg'"
        ):
            self.parse(_water_lines("Minimization: cg"))

    def test_defaults_are_those_of_parsec(self) -> None:
        """500 movements, 0.01 Ry/bohr, steps 0.05 and 1e-4 bohr, 7 corrections."""

        translation = self.parse(_water_lines("Minimization: BFGS"))
        self.assertEqual(
            translation.relaxation,
            RelaxationSettings(
                method="bfgs",
                max_movements=500,
                minimum_force=1.0e-2,
                maximum_step=0.05,
                minimum_step=1.0e-4,
                bfgs_corrections=7,
                restart=False,
            ),
        )
        self.assertEqual(translation.relaxation.force_threshold, 0.01)
        self.assertEqual(translation.relaxation.bfgs_bound, -1.0)
        self.assertEqual(translation.movable, (True, True, True))
        self.assertTrue(translation.use_symmetry)
        self.assertTrue(translation.problem.recenter_geometry)
        self.assertIsNone(translation.molecular_dynamics)

    def test_every_keyword_reaches_the_settings(self) -> None:
        """Movement_Num, Force_Min, both steps, BFGS_number_corr, Relax_Restart."""

        translation = self.parse(
            _water_lines(
                "Minimization: simple\n"
                "Movement_Num: 12\n"
                "Force_Min: 0.002\n"
                "Max_Step: 0.08\n"
                "Min_Step: 0.0004\n"
                "BFGS_number_corr: 5\n"
                "Relax_Restart: .true.\n"
            )
        )
        self.assertEqual(
            translation.relaxation,
            RelaxationSettings(
                method="simple",
                max_movements=12,
                minimum_force=0.002,
                maximum_step=0.08,
                minimum_step=0.0004,
                bfgs_corrections=5,
                restart=True,
            ),
        )

    def test_units_of_force_min_and_of_the_steps(self) -> None:
        """Ry/bohr and bohr unless a unit follows the number."""

        forces = (
            ("0.02", 0.02),
            ("0.02 Ry/bohr", 0.02),
            ("0.02 ry/au", 0.02),
            ("0.01 Hartree/bohr", 0.02),
            ("0.01 Ha/au", 0.02),
            ("0.5 eV/ang", 0.5 * EV_TO_RYDBERG / ANGSTROM_TO_BOHR),
            ("0.5 eV/angstrom", 0.5 * EV_TO_RYDBERG / ANGSTROM_TO_BOHR),
        )
        for text, expected in forces:
            with self.subTest(force_min=text):
                translation = self.parse(
                    _water_lines(f"Minimization: BFGS\nForce_Min: {text}")
                )
                self.assertAlmostEqual(
                    translation.relaxation.minimum_force, expected, delta=1e-15
                )
        # 0.5 eV/angstrom is close to 0.0194 Ry/bohr.
        self.assertAlmostEqual(
            0.5 * EV_TO_RYDBERG / ANGSTROM_TO_BOHR, 0.019447, delta=2e-6
        )
        with self.assertRaisesRegex(
            UnsupportedParsecOptionError, "unsupported force unit"
        ):
            self.parse(_water_lines("Minimization: BFGS\nForce_Min: 0.5 eV/bohr"))

        translation = self.parse(
            _water_lines(
                "Minimization: simple\nMax_Step: 0.05 ang\nMin_Step: 0.0005 bohr"
            )
        )
        self.assertAlmostEqual(
            translation.relaxation.maximum_step, 0.05 / 0.529177, delta=1e-15
        )
        self.assertEqual(translation.relaxation.minimum_step, 0.0005)
        with self.assertRaisesRegex(
            UnsupportedParsecOptionError, "unsupported length unit"
        ):
            self.parse(_water_lines("Minimization: BFGS\nMax_Step: 0.1 nm"))

    def test_max_step_at_its_default_leaves_bfgs_unrestricted(self) -> None:
        """PARSEC compares ``Max_Step`` with its default 0.05 to decide.

        A BFGS run is bounded exactly when the input gives another value;
        writing the default has no effect.
        """

        for text, bound in (
            ("0.05", -1.0), ("0.05 bohr", -1.0), ("0.08", 0.08), ("0.01", 0.01)
        ):
            with self.subTest(max_step=text):
                translation = self.parse(
                    _water_lines(f"Minimization: BFGS\nMax_Step: {text}")
                )
                self.assertEqual(translation.relaxation.bfgs_bound, bound)

    def test_values_outside_their_range_are_refused(self) -> None:
        """Limits of ``usrinputfile.F90``, and of the steepest-descent law."""

        refused = (
            (
                "Minimization: BFGS\nForce_Min: 0.000009",
                "Force_Min is below numerical accuracy",
            ),
            (
                "Minimization: manual\nForce_Min: 0.000009",
                "Force_Min is below numerical accuracy",
            ),
            ("Minimization: BFGS\nMovement_Num: -1", "Movement_Num must be nonnegative"),
            (
                "Minimization: BFGS\nBFGS_number_corr: 0",
                "BFGS_number_corr must be positive",
            ),
            (
                "Minimization: simple\nMax_Step: 0.000009\nMin_Step: 1e-7",
                "Max_Step is too small",
            ),
            # log((stepmax-stepmin)/(stepmax-50*stepmin)) needs stepmax > 50 stepmin.
            (
                "Minimization: simple\nMax_Step: 0.05\nMin_Step: 0.001",
                r"Max_Step > 50\*Min_Step > 0",
            ),
            ("Minimization: simple\nMin_Step: 0", r"Max_Step > 50\*Min_Step > 0"),
        )
        for control, message in refused:
            with self.subTest(control=control):
                with self.assertRaisesRegex(ValueError, message):
                    self.parse(_water_lines(control))
        # The same step sizes are no concern of BFGS, and no movement is allowed.
        translation = self.parse(
            _water_lines(
                "Minimization: BFGS\nMax_Step: 0.05\nMin_Step: 0.001\nMovement_Num: 0"
            )
        )
        self.assertEqual(translation.relaxation.max_movements, 0)
        self.assertEqual(
            self.parse(
                _water_lines("Minimization: BFGS\nForce_Min: 0.00001")
            ).relaxation.minimum_force,
            1e-5,
        )

    def test_move_flag_variants(self) -> None:
        """``Move_Flag`` of a species: 0 all atoms, n > 0 the first n, < 0 a column.

        A negative flag reads a fourth column per atom, 1 for movable and 0
        for fixed.
        """

        variants = (
            ("none given", {}, (True, True, True)),
            ("zero", dict(oxygen_flag="0", hydrogen_flag="0"), (True, True, True)),
            (
                "first hydrogen",
                dict(oxygen_flag="0", hydrogen_flag="1"),
                (True, True, False),
            ),
            (
                "both hydrogens",
                dict(oxygen_flag="1", hydrogen_flag="2"),
                (True, True, True),
            ),
            (
                "column",
                dict(
                    oxygen_flag="0",
                    hydrogen_flag="-1",
                    hydrogen=("1.6 0.4 0.0  0", "-1.6 0.4 0.0  1"),
                ),
                (True, False, True),
            ),
            (
                "column and count",
                dict(
                    oxygen_flag="-1",
                    oxygen=("0.0 0.0 0.0  0",),
                    hydrogen_flag="1",
                ),
                (False, True, False),
            ),
            (
                "any negative value",
                dict(oxygen_flag="-7", oxygen=("0.0 0.0 0.0  1",), hydrogen_flag="1"),
                (True, True, False),
            ),
        )
        for label, keywords, expected in variants:
            with self.subTest(case=label):
                translation = self.parse(_water_lines("Minimization: BFGS", **keywords))
                self.assertEqual(translation.movable, expected)
                self.assertEqual(len(translation.problem.atoms), 3)
        self.assertEqual(
            _translation("water_bfgs_fixed").movable,
            tuple(CASES["water_bfgs_fixed"]["movable"]),
        )

    def test_move_flag_lines_are_taken_in_their_order(self) -> None:
        """The n-th ``Move_Flag`` of the file is that of the n-th atom type.

        PARSEC's input reader keeps one cursor per keyword and hands out the
        next occurrence at every call (``kw_index`` of ``esdf_mod.f90``), so
        a single line placed in the block of the second type is read for the
        first.  Giving the flag of every type avoids the surprise.
        """

        # The fourth column is read for oxygen and not for hydrogen.
        only_in_hydrogen_block = _water_lines(
            "Minimization: BFGS",
            hydrogen_flag="-1",
            oxygen=("0.0 0.0 0.0  0",),
            hydrogen=("1.6 0.4 0.0", "-1.6 0.4 0.0  9"),
        )
        keywords = [
            line
            for line in only_in_hydrogen_block
            if line.startswith(("Atom_Type:", "Move_Flag"))
        ]
        self.assertEqual(keywords, ["Atom_Type: O", "Atom_Type: H", "Move_Flag: -1"])
        translation = self.parse(only_in_hydrogen_block)
        self.assertEqual(translation.movable, (False, True, True))

    def test_move_flag_that_does_not_fit_is_refused(self) -> None:
        """More movable atoms than atoms, a bad or missing column, none movable."""

        refused = (
            (
                dict(oxygen_flag="0", hydrogen_flag="3"),
                "Atom_Type H: invalid Move_Flag value 3",
            ),
            (dict(oxygen_flag="2"), "Atom_Type O: invalid Move_Flag value 2"),
            (
                dict(
                    oxygen_flag="0",
                    hydrogen_flag="-1",
                    hydrogen=("1.6 0.4 0.0  2", "-1.6 0.4 0.0  1"),
                ),
                "Atom_Type H: invalid atom movement index 2; use 0 or 1",
            ),
            (
                dict(oxygen_flag="0", hydrogen_flag="-1"),
                "three coordinates and a movement index",
            ),
            (
                dict(
                    oxygen_flag="-1",
                    oxygen=("0.0 0.0 0.0  0",),
                    hydrogen_flag="-1",
                    hydrogen=("1.6 0.4 0.0  0", "-1.6 0.4 0.0  0"),
                ),
                "Move_Flag leaves no atom movable",
            ),
        )
        for keywords, message in refused:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ParsecInputError, message):
                    self.parse(_water_lines("Minimization: BFGS", **keywords))

    def test_manual_moves_every_atom_and_keeps_the_coordinates(self) -> None:
        """``usrinputfile.F90`` makes all atoms movable; no symmetry, no recentring.

        The structures of ``manual.dat`` are absolute coordinates, so the
        input structure is not shifted either.
        """

        flags = dict(oxygen_flag="0", hydrogen_flag="1")
        self.assertEqual(
            self.parse(_water_lines("Minimization: BFGS", **flags)).movable,
            (True, True, False),
        )
        translation = self.parse(
            _water_lines("Minimization: manual\nMovement_Num: 2", **flags)
        )
        self.assertEqual(translation.movable, (True, True, True))
        self.assertFalse(translation.use_symmetry)
        self.assertFalse(translation.problem.recenter_geometry)
        self.assertEqual(translation.relaxation.force_threshold, 0.0)
        np.testing.assert_array_equal(
            _positions_of(translation.problem),
            [[0.0, 0.0, 0.0], [1.6, 0.4, 0.0], [-1.6, 0.4, 0.0]],
        )

    def test_keywords_without_minimization_do_nothing(self) -> None:
        """A single point: flags are not read, the controls are named as ignored.

        With a negative ``Move_Flag`` PARSEC then reads three columns only.
        """

        translation = self.parse(
            _water_lines(
                "Movement_Num: 5\nForce_Min: 0.002\nRelax_Restart: .true.",
                oxygen_flag="-1",
                hydrogen_flag="1",
            )
        )
        self.assertIsNone(translation.relaxation)
        self.assertEqual(translation.movable, (True, True, True))
        self.assertFalse(translation.compute_forces)
        warnings = "\n".join(translation.warnings)
        self.assertIn("ignored for this fixed single point", warnings)
        self.assertIn("Relax_Restart is set without Minimization", warnings)

    def test_potential_extrapolation(self) -> None:
        """``none`` (PARSEC) unless the input asks for ``atomic``; nothing else."""

        self.assertEqual(
            self.parse(_water_lines("Minimization: bfgs")).potential_extrapolation, "none"
        )
        translation = self.parse(
            _water_lines("Minimization: bfgs\nPotential_Extrapolation: Atomic")
        )
        self.assertEqual(translation.potential_extrapolation, "atomic")
        self.assertEqual(translation.warnings, self.parse(_water_lines("Minimization: bfgs")).warnings)
        single_point = self.parse(_water_lines("Potential_Extrapolation: atomic"))
        self.assertIn("no atom moves", "\n".join(single_point.warnings))
        with self.assertRaisesRegex(ParsecInputError, "Potential_Extrapolation must be one of"):
            self.parse(_water_lines("Minimization: bfgs\nPotential_Extrapolation: linear"))

    def test_periodic_relaxation_is_refused(self) -> None:
        """Relaxation is implemented for isolated systems only."""

        for scheme in ("BFGS", "simple", "manual"):
            with self.subTest(scheme=scheme):
                with self.assertRaisesRegex(
                    UnsupportedParsecOptionError,
                    "structural relaxation is implemented for isolated systems",
                ):
                    self.parse(
                        _water_lines(f"Minimization: {scheme}", domain=PERIODIC_DOMAIN)
                    )
        # The same cell without a relaxation is accepted.
        translation = self.parse(_water_lines(domain=PERIODIC_DOMAIN))
        self.assertIsNotNone(translation.problem.periodic_cell)
        self.assertIsNone(translation.relaxation)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _run_cli(arguments, main=cli_main) -> tuple[int, str, str]:
    """Exit code, standard output and standard error of a command-line driver.

    The readable driver unless ``main`` names another one.
    """

    output, error = StringIO(), StringIO()
    with redirect_stdout(output), redirect_stderr(error):
        code = main([str(argument) for argument in arguments])
    return code, output.getvalue(), error.getvalue()


class CommandLineTests(_ScratchCase):
    def test_exit_code_of_a_run(self) -> None:
        """0 done, 3 an SCF loop did not converge, 4 not relaxed.

        Done means the force criterion for a relaxation and every listed
        structure for manual movements.  An SCF failure counts before
        everything else; an optimizer that ended by itself is not relaxed.
        """

        converged = SimpleNamespace(converged=True)
        failed = SimpleNamespace(converged=False)

        def outcome(**fields):
            defaults = dict(
                scf_failed=False,
                error=0,
                minimum_force_found=False,
                done=False,
                movements_exhausted=False,
            )
            return SimpleNamespace(**{**defaults, **fields})

        self.assertEqual(relaxation_exit_code(converged, None), 0)
        self.assertEqual(relaxation_exit_code(failed, None), 3)
        self.assertEqual(
            relaxation_exit_code(converged, outcome(minimum_force_found=True)), 0
        )
        self.assertEqual(
            relaxation_exit_code(converged, outcome(movements_exhausted=True)), 4
        )
        self.assertEqual(relaxation_exit_code(converged, outcome(done=True)), 4)
        self.assertEqual(
            relaxation_exit_code(converged, outcome(done=True, error=542)), 4
        )
        self.assertEqual(relaxation_exit_code(converged, outcome()), 4)
        used_up = outcome(movements_exhausted=True)
        self.assertEqual(relaxation_exit_code(converged, used_up, manual=True), 0)
        found_but_failed = outcome(minimum_force_found=True, scf_failed=True)
        self.assertEqual(relaxation_exit_code(converged, found_but_failed), 3)
        listed_but_failed = outcome(movements_exhausted=True, scf_failed=True)
        self.assertEqual(
            relaxation_exit_code(converged, listed_but_failed, manual=True), 3
        )

    def test_dry_run_accepts_a_relaxation_input(self) -> None:
        """``--dry-run`` parses a relaxation input and writes no file."""

        source = self.write_input(
            CASES["water_coarse_bfgs_part1"]["input"], parsec_boundary=True
        )
        code, output, error = _run_cli([source, "--dry-run"])
        self.assertEqual((code, error), (0, ""))
        self.assertIn("Dry run successful", output)
        self.assertEqual(
            sorted(path.name for path in self.directory.iterdir()),
            ["H_POTRE.DAT", "O_POTRE.DAT", "parsec.in"],
        )

    def test_input_errors_end_before_any_calculation(self) -> None:
        """Exit code 2 and one line on standard error; no file is written."""

        periodic = _water_lines("Minimization: BFGS", domain=PERIODIC_DOMAIN)
        code, _, error = _run_cli([self.write_input(periodic), "--quiet"])
        self.assertEqual(code, 2)
        self.assertIn(
            "Input error: Minimization='bfgs' with Boundary_Conditions=bulk", error
        )

        source = self.write_input(CASES["water_coarse_bfgs_part1"]["input"])
        code, _, error = _run_cli([source, "--quiet", "--restart-from", "missing.dat"])
        self.assertEqual(code, 2)
        self.assertIn("Input error:", error)
        self.assertIn("missing.dat", error)

        self.write_file("two.xyz", format_xyz_frame(("O", "H"), ASYMMETRIC[:2], step=0))
        code, _, error = _run_cli([source, "--quiet", "--restart-from", "two.xyz"])
        self.assertEqual(code, 2)
        self.assertIn("Input error: two.xyz holds 2 atoms, the input 3", error)

        code, _, error = _run_cli(
            [
                self.write_input(_water_lines("Minimization: BFGS\nForce_Min: 1e-7")),
                "--quiet",
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("Input error: Force_Min is below numerical accuracy", error)
        self.assertEqual(
            sorted(path.name for path in self.directory.iterdir()),
            ["H_POTRE.DAT", "O_POTRE.DAT", "parsec.in", "two.xyz"],
        )

    def test_accelerated_driver_refuses_the_same_inputs(self) -> None:
        """The accelerated driver takes the same restart and symmetry decisions.

        These branches end with exit code 2 before a backend is chosen, so
        no accelerator is touched: a restart file for other atoms, a
        periodic relaxation, and ``--symmetry on`` where the relaxation
        cannot keep the point group.
        """

        from parsec_python.acceleration.cli import main as accelerated_main

        def refused(lines, *options) -> str:
            code, _, error = _run_cli(
                [self.write_input(lines), "--quiet", *options], main=accelerated_main
            )
            self.assertEqual(code, 2)
            return error

        relaxation = CASES["water_coarse_bfgs_part1"]["input"]
        self.write_file("two.xyz", format_xyz_frame(("O", "H"), ASYMMETRIC[:2], step=0))
        self.assertIn(
            "Input error: two.xyz holds 2 atoms, the input 3",
            refused(relaxation, "--restart-from", "two.xyz"),
        )
        self.assertIn("missing.dat", refused(relaxation, "--restart-from", "missing.dat"))
        self.assertIn(
            "structural relaxation is implemented for isolated systems",
            refused(_water_lines("Minimization: BFGS", domain=PERIODIC_DOMAIN)),
        )
        one_fixed = _water_lines("Minimization: BFGS", oxygen_flag="0", hydrogen_flag="1")
        self.assertIn(
            "--symmetry on cannot be used here: Move_Flag fixes atoms",
            refused(one_fixed, "--symmetry", "on"),
        )
        manual = _water_lines("Minimization: manual\nMovement_Num: 0")
        self.assertIn(
            "--symmetry on cannot be used here: manual movements",
            refused(manual, "--symmetry", "on"),
        )
        self.assertEqual(
            sorted(path.name for path in self.directory.iterdir()),
            ["H_POTRE.DAT", "O_POTRE.DAT", "parsec.in", "two.xyz"],
        )

    def test_manual_run_without_manual_dat_fails_before_the_first_scf_loop(self) -> None:
        """Exit code 1; the log says what is missing, and no structure is computed."""

        source = self.write_input(
            CASES["water_coarse_manual"]["input"], parsec_boundary=True
        )
        code, output, error = _run_cli([source, "--no-archive", "--quiet"])
        self.assertEqual((code, output, error), (1, "", ""))
        log = (self.directory / "parsec.out").read_text()
        self.assertIn(
            "Calculation failed: FileNotFoundError: Minimization: manual with "
            "Movement_Num 2 needs manual.dat beside the input",
            log,
        )
        self.assertNotIn("SCF iter", log)
        self.assertFalse((self.directory / "atom.cor").exists())

    def test_restart_from_and_an_scf_loop_that_does_not_converge(self) -> None:
        """One SCF iteration at the coordinates of ``--restart-from``: exit code 3.

        ``Movement_Num`` 0 computes the given structure only.  ``atom.cor``
        shows that the coordinates of a restart file of PARSEC were used as
        they are and those of an XYZ file recentred like an input, and the SCF
        failure outranks the missing relaxation in the exit code.
        """

        lines = [
            line
            for line in CASES["water_coarse_bfgs_part1"]["input"]
            if not line.startswith(("Max_Iter", "Movement_Num"))
        ] + ["Max_Iter: 1", "Movement_Num: 0"]
        source = self.write_input(lines, parsec_boundary=True)
        restart_text = CASES["water_coarse_bfgs_part1"]["files"]["relax_restart.dat"]
        expected = parse_relax_restart(restart_text, 3).positions
        elsewhere = self.directory / "earlier"
        elsewhere.mkdir()
        (elsewhere / "restart.dat").write_text(restart_text, newline="\n")
        # The XYZ file holds the structure as another program might: far from
        # the origin.  PARSEC's rule moves the middle of its bounding box there.
        elsewhere_in_space = expected + np.array([3.0, -2.0, 1.5])
        (elsewhere / "last.xyz").write_text(
            format_xyz_frame(SYMBOLS, ASYMMETRIC, step=0)
            + format_xyz_frame(SYMBOLS, elsewhere_in_space, step=1),
            newline="\n",
        )
        recentred = np.array(
            [
                atom.position
                for atom in center_cluster_geometry(
                    tuple(
                        Atom(symbol, position)
                        for symbol, position in zip(SYMBOLS, elsewhere_in_space)
                    )
                )
            ]
        )
        self.assertGreater(np.abs(recentred - elsewhere_in_space).max(), 1.0)
        np.testing.assert_allclose(
            0.5 * (recentred.min(axis=0) + recentred.max(axis=0)), 0.0, atol=1e-9
        )
        for name, expected in (("restart.dat", expected), ("last.xyz", recentred)):
            with self.subTest(source=name):
                # A relative path is taken beside the input.
                code, _, error = _run_cli(
                    [
                        source,
                        "--no-archive",
                        "--quiet",
                        "--restart-from",
                        f"earlier/{name}",
                    ]
                )
                self.assertEqual((code, error), (3, ""))
                written = _read_atom_cor((self.directory / "atom.cor").read_text())
                self.assertEqual(len(written.steps), 1)
                self.assertIsNone(written.steps[0]["energy"])
                np.testing.assert_allclose(
                    written.steps[0]["positions"], expected, rtol=0, atol=5.1e-7
                )
                np.testing.assert_allclose(written.final, expected, rtol=0, atol=5.1e-7)
                log = (self.directory / "parsec.out").read_text()
                self.assertIn(
                    "WARNING: atom coordinates in input were replaced by "
                    f"coordinates in {name}",
                    log,
                )
                self.assertEqual(log.count(" SCF iter #"), 1)
                self.assertIn("WARNING: SCF convergence not achieved", log)
                self.assertIn(" Maximum number of movements  =    0", log)
                self.assertFalse((self.directory / "relax_restart.dat").exists())
                self.assertFalse((self.directory / "parsec_python_results.npz").exists())


class DriverOnTheSpringsTests(_ScratchCase):
    """The readable driver with the springs in place of the SCF calculation.

    Parsing, the restart and symmetry decisions, the preparation of the
    system, the log, the recorder and the exit code are the driver's own;
    the four operations of the backend and the text reporter are replaced.
    """

    ASYMMETRIC_ROWS = dict(
        oxygen=("0.05 -0.12 0.08",), hydrogen=("1.55 0.45 -0.20", "-1.35 0.75 0.30")
    )

    def run_driver(self, control, surface=None, **rows):
        surface = _SurfaceBackend() if surface is None else surface
        source = self.write_input(_water_lines(control, **rows))
        with (
            patch.object(cli_module, "reference_backend", surface.backend),
            patch.object(cli_module, "ParsecTextReporter", _Reporter),
        ):
            code, output, error = _run_cli([source, "--no-archive", "--quiet"])
        self.assertEqual((output, error), ("", ""))
        return code, surface

    def log(self) -> str:
        return (self.directory / "parsec.out").read_text(encoding="utf-8")

    def statements(self) -> list[str]:
        return _read_atom_cor((self.directory / "atom.cor").read_text()).statements

    def test_relaxed_structure_is_exit_code_zero(self) -> None:
        """The run ends on ``Force_Min`` and says so in every place."""

        code, surface = self.run_driver(
            "Minimization: BFGS\nForce_Min: 0.005", **self.ASYMMETRIC_ROWS
        )
        self.assertEqual(code, 0)
        # The driver centred the input structure before the first SCF loop.
        np.testing.assert_allclose(
            surface.scf_calls[0][0], ASYMMETRIC, rtol=0, atol=1e-14
        )
        self.assertEqual(self.statements(), [MINIMUM_FOUND])
        written = _read_atom_cor((self.directory / "atom.cor").read_text())
        self.assertEqual(len(written.steps), len(surface.scf_calls))
        self.assertLess(written.steps[-1]["force"], 0.005)
        log = self.log()
        self.assertIn(" Limited-memory BFGS minimization!", log)
        self.assertIn("  Minimum force is satisfied ", log)
        self.assertIn(" Total Python wall time [sec] :", log)
        self.assertFalse((self.directory / "parsec_python_results.npz").exists())

    def test_run_that_is_not_relaxed_is_exit_code_four(self) -> None:
        """Movements used up, or an optimizer that ended at its bounds."""

        control = "Minimization: BFGS\nForce_Min: 0.005"
        code, surface = self.run_driver(
            control + "\nMovement_Num: 1", **self.ASYMMETRIC_ROWS
        )
        self.assertEqual(code, 4)
        self.assertEqual(len(surface.scf_calls), 2)
        self.assertEqual(self.statements(), [MOVEMENTS_USED])
        self.assertIn("  WARNING: Minimum force is NOT satisfied!!", self.log())

        code, _ = self.run_driver(control + "\nMax_Step: 0.02", **self.ASYMMETRIC_ROWS)
        self.assertEqual(code, 4)
        self.assertEqual(
            self.statements(),
            [" Local minimum of total energy was found", " LBFGS is terminating"],
        )
        log = self.log()
        self.assertIn(" Restricted BFGS", log)
        self.assertIn(" LBFGS minimization converged ", log)
        self.assertIn("  WARNING: Minimum force is NOT satisfied!!", log)

    def test_manual_run_is_exit_code_zero(self) -> None:
        """Every listed structure computed counts as done, whatever the forces.

        The driver leaves the coordinates of the input where they are and
        asks for no symmetry.
        """

        self.write_file("manual.dat", CASES["water_manual"]["files"]["manual.dat"])
        code, surface = self.run_driver(
            "Minimization: manual\nMovement_Num: 2", **self.ASYMMETRIC_ROWS
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(surface.scf_calls), 3)
        for index, (positions, _) in enumerate(surface.scf_calls):
            np.testing.assert_array_equal(
                positions, _array("water_manual", index, "positions")
            )
        self.assertEqual(surface.force_calls, [(False, None)] * 3)
        self.assertEqual(self.statements(), [MOVEMENTS_USED])
        self.assertNotIn("Minimum force is", self.log())

    def test_scf_failure_is_exit_code_three(self) -> None:
        """Also when the last structure meets the force criterion."""

        code, _ = self.run_driver(
            "Minimization: BFGS\nForce_Min: 0.005",
            _SurfaceBackend(unconverged=(1,)),
            **self.ASYMMETRIC_ROWS,
        )
        self.assertEqual(code, 3)
        self.assertEqual(self.statements(), [MINIMUM_FOUND])
        self.assertIn("  WARNING: SCF convergence not achieved", self.log())

    def test_energy_that_is_not_finite_fails_the_run(self) -> None:
        """Exit code 1; the log names the structure, and nothing moves from it."""

        code, surface = self.run_driver(
            "Minimization: BFGS\nForce_Min: 0.005",
            _SurfaceBackend(energies={1: float("nan")}),
            **self.ASYMMETRIC_ROWS,
        )
        self.assertEqual(code, 1)
        self.assertEqual(len(surface.scf_calls), 2)
        self.assertIn(
            "Calculation failed: FloatingPointError: the total energy or a force "
            "of structure 1 is not finite",
            self.log(),
        )
        written = _read_atom_cor((self.directory / "atom.cor").read_text())
        self.assertEqual([block["step"] for block in written.steps], [0, 1])
        self.assertIsNone(written.steps[1]["energy"])
        self.assertIsNone(written.final)

    def test_symmetry_and_movable_flags_reach_the_loop(self) -> None:
        """The decisions of the driver, as the backend sees them.

        The symmetric molecule is relaxed with its four operations unless
        the input ignores symmetry or fixes one hydrogen atom, which then
        stays where the centring of the input put it.
        """

        control = "Minimization: BFGS\nForce_Min: 0.005\nMovement_Num: 3"
        _, surface = self.run_driver(control)
        np.testing.assert_allclose(surface.scf_calls[0][0], SYMMETRIC, rtol=0, atol=1e-14)
        symmetrize, operations = surface.force_calls[0]
        self.assertTrue(symmetrize)
        self.assertEqual(len(operations), 4)

        self.assertEqual(surface.extrapolation, "none")
        _, surface = self.run_driver(control + "\nPotential_Extrapolation: atomic")
        self.assertEqual(surface.extrapolation, "atomic")
        self.assertIn(" Potential extrapolation after a move: atomic", self.log())

        _, surface = self.run_driver(control + "\nIgnore_Symmetry: .true.")
        self.assertEqual(surface.force_calls[0], (False, None))

        _, surface = self.run_driver(control, oxygen_flag="0", hydrogen_flag="1")
        self.assertEqual(surface.force_calls[0], (False, None))
        log = self.log()
        self.assertIn("WARNING: Move_Flag fixes atoms that a symmetry operation", log)
        self.assertIn(" Movable atoms =      2 of      3", log)
        self.assertEqual(len(surface.scf_calls), 4)
        first, last = surface.scf_calls[0][0], surface.scf_calls[-1][0]
        for positions, _ in surface.scf_calls:
            np.testing.assert_array_equal(positions[2], first[2])
        self.assertTrue(np.all(np.max(np.abs(last[:2] - first[:2]), axis=1) > 0.01))

    def test_run_continued_with_relax_restart(self) -> None:
        """Stop after two movements, continue with ``Relax_Restart``.

        The driver reads its own ``relax_restart.dat`` beside the input,
        does not centre the structure again, and computes to the bit the
        structures of the run that was not stopped.
        """

        control = "Minimization: BFGS\nForce_Min: 0.001\nMovement_Num: {}"
        code, whole = self.run_driver(control.format(40), **self.ASYMMETRIC_ROWS)
        self.assertEqual(code, 0)
        code, first = self.run_driver(control.format(2), **self.ASYMMETRIC_ROWS)
        self.assertEqual(code, 4)
        code, second = self.run_driver(
            control.format(37) + "\nRelax_Restart: .true.", **self.ASYMMETRIC_ROWS
        )
        self.assertEqual(code, 0)
        joined = first.scf_calls + second.scf_calls
        self.assertEqual(len(joined), len(whole.scf_calls))
        self.assertGreater(len(second.scf_calls), 3)
        for (mine, _), (theirs, _) in zip(joined, whole.scf_calls):
            np.testing.assert_array_equal(mine, theirs)
        log = self.log()
        self.assertIn(" RESTART LBFGS FROM PREVIOUS RUN", log)
        self.assertIn(" Restart relaxation, information on previous", log)


class _CommandLineRuns(unittest.TestCase):
    """Complete relaxations through the readable driver against Fortran PARSEC.

    Every case of ``NAMES`` is run once, in its own directory, as a user
    would run it: ``parsec.in`` with the pseudopotentials, ``manual.dat`` or
    the ``relax_restart.dat`` the Fortran run started from, and
    ``Hartree_Boundary_Tolerance: off`` for PARSEC's Hartree boundary
    values.
    """

    NAMES: tuple[str, ...] = ()

    @classmethod
    def setUpClass(cls) -> None:
        scratch = tempfile.TemporaryDirectory()
        cls.addClassCleanup(scratch.cleanup)
        cls.runs = {}
        for name in cls.NAMES:
            case = CASES[name]
            directory = Path(scratch.name) / name
            directory.mkdir()
            source = write_case(case, directory)
            if "manual.dat" in case["files"]:
                (directory / "manual.dat").write_text(
                    case["files"]["manual.dat"], newline="\n"
                )
            if "restart_from" in case:
                restart_text = CASES[case["restart_from"]]["files"]["relax_restart.dat"]
                (directory / "relax_restart.dat").write_text(restart_text, newline="\n")
            code, _, error = _run_cli([source, "--no-archive", "--quiet"])
            cls.runs[name] = SimpleNamespace(code=code, error=error, directory=directory)

    def read(self, name: str, file_name: str) -> str:
        return (self.runs[name].directory / file_name).read_bytes().decode("ascii")

    def assert_follows_fortran(self, name: str, *, first_block: bool = True):
        """Compare ``atom.cor`` of a run with the Fortran file, number by number."""

        self.assertEqual(self.runs[name].error, "")
        written = _read_atom_cor(self.read(name, "atom.cor"))
        reference = _read_atom_cor(CASES[name]["files"]["atom.cor"])
        self.assertEqual(len(written.steps), len(reference.steps))
        for mine, theirs in zip(written.steps, reference.steps):
            self.assertEqual(mine["step"], theirs["step"])
            self.assertEqual(mine["symbols"], theirs["symbols"])
            self.assertAlmostEqual(mine["energy"], theirs["energy"], delta=ENERGY_PRINT)
            self.assertAlmostEqual(mine["force"], theirs["force"], delta=FORCE_PRINT)
            if first_block or mine["step"] > 0:
                np.testing.assert_allclose(
                    mine["positions"], theirs["positions"], rtol=0, atol=POSITION_PRINT
                )
        self.assertEqual(written.statements, reference.statements)
        np.testing.assert_allclose(
            written.final, reference.final, rtol=0, atol=POSITION_PRINT
        )
        frames = parse_xyz_frames(self.read(name, "relaxation.xyz"))
        self.assertEqual(len(frames), len(reference.steps))
        return written, reference

    def check_bfgs_stopped_by_movement_num(self, name: str) -> None:
        self.assertEqual(self.runs[name].code, 4)
        written, _ = self.assert_follows_fortran(name)
        self.assertEqual(written.statements, [MOVEMENTS_USED])
        self.assertEqual(
            self.read(name, "iterate.dat"), CASES[name]["files"]["iterate.dat"]
        )
        ours = parse_relax_restart(self.read(name, "relax_restart.dat"), 3)
        theirs = parse_relax_restart(CASES[name]["files"]["relax_restart.dat"], 3)
        np.testing.assert_allclose(
            ours.positions, theirs.positions, rtol=0, atol=POSITION_PRINT
        )
        np.testing.assert_allclose(ours.positions, written.final, rtol=0, atol=5.1e-7)
        self.assertEqual(
            (ours.iteration, ours.movable_atoms),
            (theirs.iteration, sum(CASES[name]["movable"])),
        )
        self.assertEqual(ours.lbfgsb.task, theirs.lbfgsb.task)
        self.assertEqual(
            ours.lbfgsb.function_evaluations, theirs.lbfgsb.function_evaluations
        )
        log = self.read(name, "parsec.out")
        self.assertIn(" Limited-memory BFGS minimization!", log)
        self.assertIn(" NOTICE: Maximum number of movements used.", log)
        self.assertIn("  WARNING: Minimum force is NOT satisfied!!", log)
        self.assertNotIn("SCF convergence not achieved", log)
        self.assertFalse(
            (self.runs[name].directory / "parsec_python_results.npz").exists()
        )

    def check_manual(self, name: str) -> None:
        self.assertEqual(self.runs[name].code, 0)
        written, _ = self.assert_follows_fortran(name)
        self.assertEqual(written.statements, [MOVEMENTS_USED])
        listed = parse_manual_structures(CASES[name]["files"]["manual.dat"], 3, 2)
        restart = parse_relax_restart(self.read(name, "relax_restart.dat"), 3)
        np.testing.assert_array_equal(restart.positions, listed[1])
        self.assertEqual(restart.iteration, 2)
        self.assertIsNone(restart.lbfgsb)
        log = self.read(name, "parsec.out")
        self.assertIn(" atoms move according to info in manual.dat", log)
        self.assertNotIn("Minimum force is", log)
        self.assertFalse((self.runs[name].directory / "iterate.dat").exists())


class ReadableDriverTests(_CommandLineRuns):
    """Four runs on the coarse grid, under a minute together."""

    NAMES = (
        "water_coarse_bfgs_part1",
        "water_coarse_bfgs_part2",
        "water_coarse_manual",
        "water_coarse_bfgs_symmetric_fixed",
    )

    def test_bfgs_run_stopped_by_movement_num(self) -> None:
        """Three structures of a BFGS run without symmetry; exit code 4.

        Energies per atom, largest forces and coordinates are those of
        ``atom.cor`` of PARSEC, ``iterate.dat`` is identical, and the
        restart file holds PARSEC's next structure and optimizer task.
        """

        self.check_bfgs_stopped_by_movement_num("water_coarse_bfgs_part1")

    def test_bfgs_run_continued_from_the_restart_file_of_parsec(self) -> None:
        """``Relax_Restart`` with the file Fortran PARSEC wrote; exit code 4.

        The two structures of the continuation follow PARSEC's.  ``Step #
        0`` of ``atom.cor`` is the restart structure, where PARSEC writes
        the coordinates of the input.
        """

        name = "water_coarse_bfgs_part2"
        self.assertEqual(self.runs[name].code, 4)
        written, reference = self.assert_follows_fortran(name, first_block=False)
        restart = parse_relax_restart(
            CASES["water_coarse_bfgs_part1"]["files"]["relax_restart.dat"], 3
        )
        np.testing.assert_allclose(
            written.steps[0]["positions"], restart.positions, rtol=0, atol=5.1e-7
        )
        np.testing.assert_array_equal(reference.steps[0]["positions"], ASYMMETRIC)
        log = self.read(name, "parsec.out")
        self.assertIn(
            "WARNING: atom coordinates in input were replaced by coordinates "
            "in relax_restart.dat",
            log,
        )
        self.assertIn(" RESTART LBFGS FROM PREVIOUS RUN", log)
        self.assertIn(" runs read from relax_restart.dat file.", log)
        # The file was rewritten for the next continuation.
        ours = parse_relax_restart(self.read(name, "relax_restart.dat"), 3)
        np.testing.assert_allclose(ours.positions, written.final, rtol=0, atol=5.1e-7)
        self.assertEqual(ours.iteration, 1)

    def test_manual_run(self) -> None:
        """The input structure and the two of ``manual.dat``; exit code 0."""

        self.check_manual("water_coarse_manual")

    def test_symmetric_run_with_a_fixed_atom(self) -> None:
        """Oxygen fixed by ``Move_Flag``, the hydrogen atoms mirror images.

        The four operations of the molecule respect the movable flags, so
        the run keeps them: the two structures are those of PARSEC, the
        fixed atom stays at its centred input position, and the structure
        the optimizer asks for next is symmetric to round-off.  Exit code 4.
        """

        name = "water_coarse_bfgs_symmetric_fixed"
        self.check_bfgs_stopped_by_movement_num(name)
        restart = parse_relax_restart(self.read(name, "relax_restart.dat"), 3)
        self.assertEqual(restart.lbfgsb.n, 6)
        np.testing.assert_array_equal(restart.positions[0], SYMMETRIC[0])
        self.assertLess(_asymmetry(restart.positions), 1e-13)
        self.assertGreater(np.max(np.abs(restart.positions[1:] - SYMMETRIC[1:])), 1.0)
        log = self.read(name, "parsec.out")
        self.assertIn(" Movable atoms =      2 of      3", log)
        self.assertIn("  symmetrizing forces according to symmetry operations ", log)
        self.assertNotIn("Move_Flag fixes atoms", log)


@unittest.skipUnless(
    FULL_SIZE,
    "set PARSEC_TEST_RELAXATION_FULL=1 for the reference grid (about 10 minutes)",
)
class ReadableDriverFullSizeTests(_CommandLineRuns):
    """The same comparisons on the grid of the references (7.5 bohr, 0.4 bohr)."""

    NAMES = ("water_bfgs_part1", "water_manual")

    def test_bfgs_run_stopped_by_movement_num(self) -> None:
        """``water_bfgs_part1``: three structures on the grid of the references."""

        self.check_bfgs_stopped_by_movement_num("water_bfgs_part1")

    def test_manual_run(self) -> None:
        """``water_manual``: three structures on the grid of the references."""

        self.check_manual("water_manual")


if __name__ == "__main__":
    unittest.main()
