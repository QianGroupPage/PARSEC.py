"""The correctness of the sectors of the k-points (``acceleration/SCF/kpoint_sectors.py``) and the refit of their
states.

* The fixed occupations of ``occup.in`` stay out of the sectors of the spin-polarized k-point loop, as out of the
  scalar loop.
* The symmetry guard of a spin-polarized loop in the sectors (``PARSEC_KPOINT_SECTOR_GUARD``), the analogue of the
  guard of ``Spin_Symmetry_Sectors`` on clusters: its three tests on made-up levels and densities, and the oxygen
  atom in a periodic cube at the Gamma point (a partly filled minority p shell across three sectors), which leaves
  the sectors after the start Hamiltonian and is then the loop with one task per k-point, bit for bit.  What the
  tests measured goes to ``timings.detail`` and ``parsec.out`` whether they pass (a spin-polarized silicon cube at
  the Gamma point) or not (since 2026-11-19).
* The refit of the states of the sectors after the first solve (``PARSEC_KPOINT_SECTOR_REFIT``): the Ritz states cut
  to their lowest pairs (host and device kinds), and the SCF of a mesh with the refit gives the energy without it.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np

import parsec_python.acceleration  # noqa: F401  (installs the k-point solver)
from parsec_python.SCF.kpoint_solver import KPointJob, SerialKPointSolver
from parsec_python.acceleration.SCF import kpoint_sectors as sectors

_OXYGEN_INPUT = """\
Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  9.9 9.9 9.9
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: 0.55
Expansion_Order: 8
States_Num: 24
Fermi_Temp: {temperature}
Max_Iter: {max_iter}
Convergence_Criterion: {criterion}
Diag_Tolerance: 1.0e-8
Eigensolver: chebff
Correlation_Type: ca
Spin_Polarization: .true.
Atom_Types_Num: 1
Atom_Type: O
Local_Component: p
begin Atom_Coord
{coordinates}end Atom_Coord
"""


def _oxygen(
    temperature: str = "80",
    occup: list[str] | None = None,
    *,
    body_centre: bool = False,
    criterion: str = "1.0e-7",
    max_iter: int = 120,
):
    """The O atom at the origin of a periodic cube of 9.9 bohr (18 points a side): Oh about it, at the Gamma point
    the eight commuting involutions of D2h hold p_x, p_y and p_z in three sectors.  ``body_centre``: a second O atom
    at the centre of the cube (the projectors of two atoms are dealt over the ranks of a pool of two)."""

    from parsec_python.Input import parse_parsec_input
    from parsec_python.tests.spin_reference import pseudopotential_file

    directory = Path(tempfile.mkdtemp(prefix="parsec_kguard_"))
    coordinates = "   0.000000   0.000000   0.000000\n" + ("   4.950000   4.950000   4.950000\n" if body_centre else "")
    (directory / "parsec.in").write_text(
        _OXYGEN_INPUT.format(temperature=temperature, coordinates=coordinates, criterion=criterion, max_iter=max_iter)
    )
    shutil.copyfile(pseudopotential_file("tests/data/spin_ca_O_POTRE.DAT"), directory / "O_POTRE.DAT")
    if occup is not None:
        (directory / "occup.in").write_text("\n".join(occup) + "\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = parse_parsec_input(directory / "parsec.in").problem
    return directory, problem


def setUpModule() -> None:
    # the cells of these checks are small: split them all the same
    patcher = mock.patch.object(sectors, "_MINIMUM_ENTRIES", 0)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


class FixedOccupationTests(unittest.TestCase):
    def test_occup_in_keeps_the_spin_loop_out_of_the_sectors(self) -> None:
        # the job the spin loop gives its solver: no geometry (no sectors) with the occupations of occup.in
        from parsec_python.SCF import kpoints
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.tests.spin_reference import load_cases

        class Stop(Exception):
            pass

        def capture(job, **options):
            seen.append(job.geometry)
            raise Stop

        for temperature, occup in (("-1", load_cases()["o_spherical_ca"]["occup"]), ("80", None)):
            directory, problem = _oxygen(temperature, occup)
            self.addCleanup(shutil.rmtree, directory, True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                system = prepare_periodic_single_point(problem)
            seen: list = []
            with mock.patch.object(kpoints, "make_kpoint_solver", capture), self.assertRaises(Stop):
                kpoints.run_scf_kpoints_spin_polarized(system, np.zeros((1, 3)), np.ones(1))
            if occup is not None:
                self.assertIsNotNone(problem.scf.fixed_occupations)
                self.assertEqual(seen, [None])
            else:
                self.assertIsNotNone(seen[0])


class _Plan:
    """A stand-in for the sectors of one task (what the guard reads of them)."""

    def __init__(self, order: int, assumed: bool = True, group=None) -> None:
        self.order = order
        self.assumed = assumed
        self.guard_group = group
        self.sizes = (10,) * order
        self.counts = [5] * order


def _guard_solver(plans, weights, kept, k_of_task=None):
    solver = object.__new__(sectors.SectorKPointSolver)
    geometry = {"weights": np.asarray(weights, dtype=float)}
    if k_of_task is not None:
        geometry["k_of_task"] = np.asarray(k_of_task)
    solver.__dict__.update(
        {
            "job": KPointJob(len(plans), 3, lambda *a: None, lambda *a: None, lambda *a, **k: {}, geometry=geometry),
            "_plans": plans,
            "_kept": kept,
            "_inner": None,
        }
    )
    return solver


class GuardTestsOnMadeUpLevels(unittest.TestCase):
    def test_a_partly_filled_multiplet_across_sectors(self) -> None:
        # three degenerate levels in three sectors with one electron: reported at the Gamma point (weight 1),
        # not with that k-point's weight 1/27 (w_k min(n, d - n) below 0.05), nor where they are full
        levels = [np.array([-1.0, -0.5, -0.5, -0.5, 0.2])]
        owners = np.array([0, 1, 2, 3, 0])
        kept = {0: (owners, np.arange(5))}
        filled = [np.array([1.0, 1.0 / 3, 1.0 / 3, 1.0 / 3, 0.0])]
        report = _guard_solver([_Plan(4)], [1.0], kept).start_spectrum_report(levels, filled)
        self.assertIsNotNone(report)
        self.assertAlmostEqual(report.largest, 1.0, delta=1e-12)
        self.assertIn("3 levels", report.lines[0])
        self.assertIsNone(_guard_solver([_Plan(4)], [1.0 / 27], kept).start_spectrum_report(levels, filled))
        full = [np.array([1.0, 1.0, 1.0, 1.0, 0.0])]
        self.assertIsNone(_guard_solver([_Plan(4)], [1.0], kept).start_spectrum_report(levels, full))
        # what test 1 measured, also where it passes (timings.detail and parsec.out, since 2026-11-19)
        self.assertEqual(_guard_solver([_Plan(4)], [1.0], kept).start_spectrum_measure(levels, filled), report.largest)
        self.assertAlmostEqual(
            _guard_solver([_Plan(4)], [1.0 / 27], kept).start_spectrum_measure(levels, filled), 1.0 / 27, delta=1e-15
        )
        self.assertEqual(_guard_solver([_Plan(4)], [1.0], kept).start_spectrum_measure(levels, full), 0.0)
        # partners within one sector mix there freely: not reported; nor on a task whose sectors assume nothing
        one = {0: (np.array([0, 1, 1, 1, 0]), np.arange(5))}
        self.assertIsNone(_guard_solver([_Plan(4)], [1.0], one).start_spectrum_report(levels, filled))
        self.assertIsNone(_guard_solver([_Plan(4, assumed=False)], [1.0], kept).start_spectrum_report(levels, filled))
        self.assertEqual(_guard_solver([_Plan(4, assumed=False), None], [1.0], {}).guarded_tasks(), [])

    def test_fractional_occupations_at_the_end(self) -> None:
        levels = [np.array([-1.0, -0.5, -0.5, -0.5, 0.2]), np.array([-1.0, -0.6, -0.4, 0.1, 0.2])]
        kept = {task: (np.array([0, 1, 2, 3, 0]), np.arange(5)) for task in range(2)}
        filled = [np.array([1.0, 1.0 / 3, 1.0 / 3, 1.0 / 3, 0.0]), np.array([1.0, 1.0, 0.0, 0.0, 0.0])]
        solver = _guard_solver([_Plan(4), _Plan(4)], [1.0], kept, k_of_task=[0, 0])
        report = solver.fractional_report(levels, filled)
        self.assertIsNotNone(report)
        self.assertEqual(len(report.lines), 1)
        self.assertTrue(report.lines[0].startswith("task 0"))
        integer = [np.array([1.0, 1.0, 0.0, 0.0, 0.0])] * 2
        self.assertIsNone(solver.fractional_report(levels, integer))
        # what test 3 measured, also where it passes
        self.assertEqual(solver.fractional_measure(levels, filled), report.largest)
        self.assertEqual(solver.fractional_measure(levels, integer), 0.0)
        light = _guard_solver([_Plan(4), _Plan(4)], [1.0 / 27], kept, k_of_task=[0, 0])
        self.assertIsNone(light.fractional_report(levels, filled))
        self.assertAlmostEqual(light.fractional_measure(levels, filled), report.largest / 27, delta=1e-15)

    def test_the_densities_against_the_operations_of_the_cell(self) -> None:
        # the silicon cube at the Gamma point alone, no symmetrized density: every operation of the cell is assumed,
        # and a density that breaks them is measured (a symmetric one is not)
        from parsec_python.acceleration.tests.test_kpoint_sectors import _silicon_system

        problem, system = _silicon_system()
        geometry = {
            "grid": system.grid,
            "atoms": tuple(system.cell_atoms),
            "k_points": np.zeros((1, 3)),
            "weights": np.ones(1),
            "lattice": np.asarray(problem.periodic_cell.lattice_vectors, dtype=float),
            "density_symmetry": None,
        }
        job = KPointJob(1, 22, lambda *a: None, lambda *a: None, lambda *a, **k: {}, geometry=geometry)
        solver = sectors.sector_solver(job, lambda inner, **options: SerialKPointSolver(inner), problem_size=system.grid.size)
        self.assertIsInstance(solver, sectors.SectorKPointSolver)
        self.assertEqual(solver.guarded_tasks(), [0])
        images = solver._plans[0].guard_group.images()
        self.assertGreater(len(images), 0)
        self.assertLess(len(images), 8)  # a generating set, not the 192 operations
        volume = system.grid.volume_element
        symmetric = np.asarray(system.initial_density, dtype=float)
        self.assertLess(solver.density_symmetry_measure((symmetric, symmetric), volume), 1.0e-8)
        self.assertIsNone(solver.density_symmetry_report((symmetric, symmetric), volume))
        broken = symmetric.copy()
        broken[np.argmax(broken)] += 1.0 / volume  # one electron more at one point
        self.assertGreater(solver.density_symmetry_measure((symmetric, broken), volume), 1.0)
        report = solver.density_symmetry_report((symmetric, broken), volume)
        self.assertIsNotNone(report)
        self.assertTrue(all(line.startswith("channel 2") for line in report.lines))


class GuardRunTests(unittest.TestCase):
    """The oxygen atom in a periodic cube at the Gamma point (spin-polarized, 80 K)."""

    def run_oxygen(self, environment, callback=None):
        from parsec_python.SCF.kpoints import run_scf_kpoints_spin_polarized
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        directory, problem = _oxygen("80")
        self.addCleanup(shutil.rmtree, directory, True)
        records: list = []
        with mock.patch.dict(os.environ, environment), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = prepare_periodic_single_point(problem)
            result = run_scf_kpoints_spin_polarized(
                system, np.zeros((1, 3)), np.ones(1), callback=records.append if callback is None else callback
            )
        return result, records

    def test_the_reporter_of_the_command_line_writes_the_note(self) -> None:
        # the reference command line reports a spin k-point loop with iteration_spin_polarized (of parsec.out, with
        # the data files): it receives the note of the guard first, then the iterations of the loop without sectors
        from parsec_python.Input import parse_parsec_input
        from parsec_python.Output.output_files import ReporterWithFiles, RunOutputFiles
        from parsec_python.Output.parsec_output import ParsecTextReporter

        directory, _ = _oxygen("80")
        self.addCleanup(shutil.rmtree, directory, True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            translation = parse_parsec_input(directory / "parsec.in")
        lines: list[str] = []
        files = RunOutputFiles(translation.output_files, directory, write=lambda _line: None, kpoint_count=1)
        reporter = ReporterWithFiles(ParsecTextReporter(lines.append, translation), files)
        result, _ = self.run_oxygen({"PARSEC_KPOINT_SECTORS": "auto"}, callback=reporter.iteration_spin_polarized)
        self.assertEqual(result.timings.detail.get("kpoint_sector_abandoned"), 1.0)
        text = "\n".join(lines)
        self.assertEqual(text.count("NOTE: k-point sectors: a partly filled multiplet after the start Hamiltonian"), 1)
        self.assertLess(text.index("NOTE: k-point sectors"), text.index("Spin-polarized SCF iter #   1"))
        self.assertEqual(text.count("Spin-polarized SCF iter #"), result.iterations)
        # the record alone: the same lines through iteration (the ionic loops and hybrids) and iteration_spin_polarized
        from parsec_python.models import SpinSectorFallbackRecord

        record = SpinSectorFallbackRecord(
            stage="the end", largest_fractional_occupation=0.5, tolerance=0.05, multiplets=("task 0",),
            reason="a fractional occupation", measure="largest w_k min(f, 1 - f)", route="kpoint",
        )
        written: dict[str, list[str]] = {"iteration": [], "iteration_spin_polarized": []}
        for name, out in written.items():
            getattr(ReporterWithFiles(ParsecTextReporter(out.append, translation), files), name)(record)
        self.assertEqual(written["iteration_spin_polarized"], written["iteration"])
        self.assertIn("NOTE: k-point sectors: a fractional occupation after the end", written["iteration"][0])

    def test_a_partly_filled_p_shell_leaves_the_sectors(self) -> None:
        from parsec_python.Output.parsec_output import spin_sector_fallback_lines
        from parsec_python.models import SpinSectorFallbackRecord

        whole, _ = self.run_oxygen({"PARSEC_KPOINT_SECTORS": "off"})
        result, records = self.run_oxygen({"PARSEC_KPOINT_SECTORS": "auto"})
        notes = [record for record in records if isinstance(record, SpinSectorFallbackRecord)]
        self.assertEqual(len(notes), 1)
        self.assertEqual((notes[0].stage, notes[0].route), ("the start Hamiltonian", "kpoint"))
        self.assertEqual(notes[0].reason, "a partly filled multiplet")
        self.assertIn("3 levels", notes[0].multiplets[0])
        self.assertIn("one task per k-point", "\n".join(spin_sector_fallback_lines(notes[0])))
        # the loop with one task per k-point (the threads of the pool sum in their own order: to round-off)
        self.assertTrue(result.converged)
        self.assertEqual(result.timings.detail.get("kpoint_sector_abandoned"), 1.0)
        self.assertNotIn("kpoint_sector_tasks", result.timings.detail)
        # the loop without sectors carries what test 1 measured in the sectors it left, and parsec.out prints it
        from parsec_python.Output.parsec_output import ParsecTextReporter

        self.assertEqual(result.timings.detail["kpoint_guard_start_share"], notes[0].largest_fractional_occupation)
        self.assertNotIn("kpoint_guard_density_asymmetry", result.timings.detail)
        lines = ParsecTextReporter._kpoint_detail_lines(result.timings)
        self.assertIn(" K-point sector guard (measured, tolerance; in the sectors the loop left)", lines)
        self.assertTrue(any(line.lstrip().startswith("Test 1, start multiplets") for line in lines), lines)
        self.assertAlmostEqual(result.energies.total, whole.energies.total, delta=1.0e-9)
        np.testing.assert_allclose(result.eigenvalues_down, whole.eigenvalues_down, rtol=0.0, atol=1.0e-8)

    def test_the_former_route_without_the_guard(self) -> None:
        # the sectors kept: the symmetric ensemble of the p shell, above the state the loop without sectors finds
        from parsec_python.models import SpinSectorFallbackRecord

        whole, _ = self.run_oxygen({"PARSEC_KPOINT_SECTORS": "off"})
        result, records = self.run_oxygen({"PARSEC_KPOINT_SECTORS": "auto", "PARSEC_KPOINT_SECTOR_GUARD": "off"})
        self.assertFalse([record for record in records if isinstance(record, SpinSectorFallbackRecord)])
        self.assertGreater(result.timings.detail["kpoint_sector_tasks"], 2.0)
        self.assertNotIn("kpoint_sector_abandoned", result.timings.detail)
        self.assertFalse([key for key in result.timings.detail if key.startswith("kpoint_guard_")])
        self.assertGreater(result.energies.total - whole.energies.total, 1.0e-3)

    def test_a_guard_that_passes_reports_what_it_measured(self) -> None:
        # The silicon cube at the Gamma point, spin-polarized without a moment: the guard keeps the sectors, and
        # timings.detail and parsec.out hold the values of its three tests below their tolerances (2026-11-19).
        from parsec_python.Output.parsec_output import ParsecTextReporter
        from parsec_python.SCF.kpoints import run_scf_kpoints_spin_polarized
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.acceleration.SCF.spin_sectors import (
            DENSITY_SYMMETRY_TOLERANCE,
            SPIN_SECTOR_OCCUPATION_TOLERANCE,
        )
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon

        problem = silicon("ca", spacing=0.855)
        problem = replace(problem, scf=replace(problem.scf, spin_polarized=True, convergence_criterion=1.0e-6))
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTORS": "auto"}), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = prepare_periodic_single_point(problem)
            result = run_scf_kpoints_spin_polarized(system, np.zeros((1, 3)), np.ones(1))
        detail = result.timings.detail
        self.assertTrue(result.converged)
        self.assertGreater(detail["kpoint_sector_tasks"], 2.0)
        self.assertNotIn("kpoint_sector_abandoned", detail)
        self.assertLessEqual(detail["kpoint_guard_start_share"], SPIN_SECTOR_OCCUPATION_TOLERANCE)
        self.assertEqual(detail["kpoint_guard_start_passes"], 0.0)
        self.assertLess(detail["kpoint_guard_density_asymmetry"], DENSITY_SYMMETRY_TOLERANCE)
        self.assertIn("kpoint_guard_density_asymmetry_first", detail)
        self.assertEqual(detail["kpoint_guard_tightenings"], 0.0)
        self.assertLess(detail["kpoint_guard_fractional_share"], SPIN_SECTOR_OCCUPATION_TOLERANCE)
        lines = ParsecTextReporter._kpoint_detail_lines(result.timings)
        self.assertIn(" K-point sector guard (measured, tolerance)", lines)
        for label in ("Test 1, start multiplets", "Test 2, density asymmetry", "Test 3, fractional levels"):
            self.assertTrue(any(line.lstrip().startswith(label) for line in lines), (label, lines))


class RefitTests(unittest.TestCase):
    def test_the_switches(self) -> None:
        for name, function in (
            ("PARSEC_KPOINT_SECTOR_REFIT", sectors.sector_refit_setting),
            ("PARSEC_KPOINT_SECTOR_GUARD", sectors.sector_guard_setting),
        ):
            for value, expected in (("", "auto"), ("OFF", "off"), ("auto", "auto")):
                with mock.patch.dict(os.environ, {name: value}):
                    self.assertEqual(function(), expected)
            with mock.patch.dict(os.environ, {name: "on"}), self.assertRaises(ValueError):
                function()

    def test_the_refitted_counts(self) -> None:
        # the levels held plus a tenth of them, at least six more, never above the rows or the count so far
        self.assertEqual(sectors.refitted_count(200, 10_000, 400), 220)
        self.assertEqual(sectors.refitted_count(20, 10_000, 400), 26)
        self.assertEqual(sectors.refitted_count(20, 22, 400), 22)
        self.assertEqual(sectors.refitted_count(300, 10_000, 310), 310)
        self.assertEqual(sectors.refitted_count(0, 10_000, 40), 6)

    def test_a_host_state_cut_to_its_lowest_pairs(self) -> None:
        # the state of the host eigensolver: the cut state is taken by the next solve (no restart) and gives the
        # lowest levels of the operator
        from parsec_python.Eigensolvers import solve_eigval
        from parsec_python.Eigensolvers.eigval import EigvalSettings

        rng = np.random.default_rng(5)
        dimension = 300
        matrix = rng.standard_normal((dimension, dimension))
        matrix = 0.5 * (matrix + matrix.T) + np.diag(np.arange(dimension, dtype=float))
        settings = EigvalSettings()
        first = solve_eigval(matrix, 30, settings=settings)
        state = first.state
        working = 12 + settings.safety_buffer
        self.assertGreater(state.working_states, working)
        cut = sectors.trimmed_state(state, 12, working)
        self.assertEqual((cut.working_states, cut.requested_states), (working, 12))
        self.assertEqual(cut.subspace.vectors.shape, (dimension, working))
        np.testing.assert_array_equal(cut.subspace.eigenvalues, state.subspace.eigenvalues[:working])
        np.testing.assert_array_equal(cut.subspace.vectors, state.subspace.vectors[:, :working])
        second = solve_eigval(matrix, 12, settings=settings, state=cut)
        self.assertFalse(second.restarted)
        exact = np.linalg.eigvalsh(matrix)[:12]
        np.testing.assert_allclose(second.eigenvalues, exact, atol=1.0e-6)
        # carried_state cuts it the same way, and leaves it alone under PARSEC_KPOINT_SECTOR_REFIT=off
        carried = sectors.carried_state(state, 12, settings, dimension)
        self.assertEqual(carried.working_states, working)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_REFIT": "off"}):
            self.assertIs(sectors.carried_state(state, 12, settings, dimension), state)
        self.assertIs(sectors.carried_state(state, 30, settings, dimension), state)

    def test_device_kinds_of_states(self) -> None:
        # one device, and a basis shared by the columns of one rank: the first columns, in their memory order
        from parsec_python.acceleration.Eigensolvers.chebff_columns import ColumnEigvalState, ColumnLayout
        from parsec_python.acceleration.Eigensolvers.chebff_spinor import SpinorEigvalState

        vectors = np.asfortranarray(np.arange(40.0).reshape(8, 5) + 0j)
        values = np.arange(5.0)
        one = SpinorEigvalState(vectors=vectors, eigenvalues=values, working_states=5, operator_dimension=8,
                                first_filter=False, filter_lower_bound=7.0, filters_completed=3)
        cut = sectors.trimmed_state(one, 2, 3)
        np.testing.assert_array_equal(cut.vectors, vectors[:, :3])
        self.assertTrue(cut.vectors.flags.f_contiguous)
        self.assertEqual((cut.working_states, cut.filter_lower_bound, cut.filters_completed), (3, 2.0, 3))
        layout = ColumnLayout(5, 2, 1)
        shared = ColumnEigvalState(vectors=vectors, eigenvalues=values, working_states=5, operator_dimension=8,
                                   first_filter=False, filter_lower_bound=7.0, filters_completed=3, layout=layout)
        cut = sectors.trimmed_state(shared, 2, 3)
        self.assertEqual(cut.layout, ColumnLayout(3, 2, 1))
        np.testing.assert_array_equal(cut.vectors, vectors[:, :3])

    def test_a_mesh_with_the_refit_gives_the_energy_without_it(self) -> None:
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.acceleration.tests.test_kpoint_sectors import SectorSCFTests

        base = SectorSCFTests._silicon(self)
        side = float(np.asarray(base.periodic_cell.lattice_vectors)[0, 0])
        k_points = np.array([np.zeros(3), np.pi / side * np.array([1.0, 0.0, 0.0])])
        weights = np.array([0.5, 0.5])
        results = {}
        for setting in ("off", "auto"):
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_REFIT": setting}), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results[setting] = run_scf_kpoints(prepare_periodic_single_point(base), k_points, weights)
        off, auto = results["off"], results["auto"]
        self.assertTrue(off.converged and auto.converged)
        self.assertEqual(off.timings.detail["kpoint_sector_refit_states"], 0.0)
        self.assertGreater(auto.timings.detail["kpoint_sector_refit_states"], 0.0)
        self.assertLess(auto.timings.detail["kpoint_sector_work_fraction"], off.timings.detail["kpoint_sector_work_fraction"])
        self.assertAlmostEqual(auto.energies.total, off.energies.total, delta=1.0e-9)
        np.testing.assert_allclose(auto.eigenvalues, off.eigenvalues, atol=1.0e-8)


if __name__ == "__main__":
    unittest.main()
