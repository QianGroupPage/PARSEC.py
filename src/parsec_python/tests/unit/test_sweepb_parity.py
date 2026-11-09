"""Final sweep (part B) of the Fortran-parity round.

Fix_Number_Eigenvalues with restarts, Dynamic_Diag_Tol in the k-point and
spinor drivers, Save_Intermediate_* in every SCF driver, nscf_kpoints with
spin-orbit spinors, PARSEC's symmetry printouts (symchk, Abelian subgroup,
multiplication tables) and the ion-ion force table of the setup.
"""


# With both sets of routines: the periodic inputs of this module set Use_Fortran_Routines: .true. (the
# Fortran-derived routines); the numbers checked here are those of the Fortran parity round.
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from parsec_python.Input import parse_parsec_input
from parsec_python.Input.parsec_input import UnsupportedParsecOptionError
from parsec_python.models import Atom, EigensolverSettings
from parsec_python.tests.environment import kpoints_on_the_cpu, pin_kpoints_on_the_cpu

DATA = Path(__file__).resolve().parents[1] / "data"
AU_POTRE = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"

H2 = """\
Boundary_Conditions: cluster
Boundary_Sphere_Radius: 5.0 bohr
Grid_Spacing: 0.5 bohr
Coordinate_Unit: Cartesian_Bohr
Ignore_Symmetry: .true.
Hartree_Boundary_Tolerance: off
Expansion_Order: 4
States_Num: 4
Fermi_Temp: 500.0
Max_Iter: 40
Convergence_Criterion: 1.0e-6
Eigensolver: chebff
FF_Maxiter: 9
Chebdav_Degree: 40
Mixing_Param: 0.3
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
   0.0 0.0 -0.7
   0.0 0.0 0.7
end Atom_Coord
Correlation_Type: ca
{extra}
"""

CELL = """\
Periodic_System: .true.
Use_Fortran_Routines: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  6.0 6.0 6.0
end Cell_Shape
Grid_Spacing: 0.75
Expansion_Order: 4
States_Num: 4
Fermi_Temp: 500.0
Max_Iter: {iterations}
Convergence_Criterion: 1.0e-5
Eigensolver: chebdav
Diag_Tolerance: 1.0e-7
Mixing_Method: Anderson
Kpoint_Method: manual
Kpoint_Unit: reciprocal_lattice_vectors
begin Kpoint_List
  0.0 0.0 0.0 1.0
  0.25 0.0 0.0 1.0
end Kpoint_List
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  3.1 2.9 3.05
end Atom_Coord
Correlation_Type: ca
{extra}
"""

AU_CELL = """\
Periodic_System: .true.
Use_Fortran_Routines: .true.
Boundary_Conditions: bulk
begin Cell_Shape
   8.0  8.0  8.0
end Cell_Shape
Grid_Spacing: 0.6
Expansion_Order: 6
States_Num: 14
Fermi_Temp: 500.0
Max_Iter: {iterations}
Convergence_Criterion: 1.0e-5
FF_MaxIter: 9
Chebdav_Degree: 30
Mixing_Param: 0.25
Eigensolver: chebff
SO_from_scratch: .true.
Atom_Types_Num: 1
Atom_Type: Au
SO_PSP: .true.
Local_Component: s
Begin atom_coord
   4.0 4.0 4.0
End atom_coord
Correlation_Type: CA
{extra}
"""


def _write(directory: Path, template: str, **fields) -> Path:
    if "Au" in template:
        shutil.copy(AU_POTRE, directory / "Au_POTRE.DAT")
    else:
        shutil.copy(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
    path = directory / "parsec.in"
    path.write_text(template.format(**fields), encoding="utf-8")
    return path


def _parse(template: str, **fields):
    with tempfile.TemporaryDirectory() as folder:
        return parse_parsec_input(_write(Path(folder), template, **fields))


def _parse_kept(test: unittest.TestCase, template: str, **fields):
    """A translation whose pseudopotential files stay for the preparation."""

    folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    test.addCleanup(folder.cleanup)
    return parse_parsec_input(_write(Path(folder.name), template, **fields))


def _run_cli(path: Path, *extra: str) -> tuple[int, str]:
    from parsec_python.cli import main

    with kpoints_on_the_cpu(), redirect_stdout(StringIO()), redirect_stderr(StringIO()) as error:
        code = main([str(path), "--quiet", *extra])
    return code, error.getvalue()


# ----------------------------------------------------------------------------
# 1. Fix_Number_Eigenvalues
# ----------------------------------------------------------------------------


class FixNumberEigenvaluesTests(unittest.TestCase):
    def test_parser_needs_a_restart_and_drops_the_buffer(self) -> None:
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Restart_Run"):
            _parse(H2, extra="Fix_Number_Eigenvalues: .true.")
        translation = _parse(H2, extra="Restart_Run: .true.\nFix_Number_Eigenvalues: .true.")
        eigensolver = translation.problem.eigensolver
        self.assertTrue(translation.fix_number_eigenvalues)
        self.assertTrue(eigensolver.fixed_state_count)
        self.assertEqual(eigensolver.subspace_buffer, 6)
        self.assertEqual(eigensolver.working_buffer, 0)
        self.assertIn("fixed_state_count=True", repr(eigensolver))
        self.assertNotIn("fixed_state_count", repr(EigensolverSettings()))
        self.assertEqual(EigensolverSettings().working_buffer, 6)

    def test_nscf_keeps_its_own_count(self) -> None:
        translation = _parse(
            CELL,
            iterations=40,
            extra=(
                "Restart_Run: .true.\nFix_Number_Eigenvalues: .true.\n"
                "begin nscf_kpoints\n0 0 0 1\nend nscf_kpoints\n"
            ),
        )
        self.assertTrue(translation.fix_number_eigenvalues)
        self.assertFalse(translation.problem.eigensolver.fixed_state_count)

    def test_restart_file_with_fewer_levels_is_error_404(self) -> None:
        from parsec_python.SCF.restart import RestartError, check_fixed_state_count

        check_fixed_state_count(8, 8)
        check_fixed_state_count(None, 8)
        with self.assertRaises(RestartError) as caught:
            check_fixed_state_count(8, 10)
        self.assertEqual(caught.exception.code, 404)

    def test_restart_with_fixed_count_converges_to_the_same_energy(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = _write(directory, H2, extra="")
            code, error = _run_cli(path)
            self.assertEqual(code, 0, error)
            with np.load(directory / "parsec_python_results.npz") as archive:
                energy = float(archive["energy_total_ry"])
            path.write_text(
                H2.format(extra="Restart_Run: .true.\nFix_Number_Eigenvalues: .true."),
                encoding="utf-8",
            )
            code, error = _run_cli(path)
            self.assertEqual(code, 0, error)
            with np.load(directory / "parsec_python_results.npz") as archive:
                self.assertAlmostEqual(float(archive["energy_total_ry"]), energy, places=6)
            log = (directory / "parsec.out").read_text()
            self.assertIn(" Fixing the number of eigenvalues per representation", log)
            self.assertIn(
                " WARNING!! This option should only be used when restarting from previous SCF.",
                log,
            )
            # The stored States_Num is 4: asking for 6 is PARSEC's error 404.
            path.write_text(
                H2.format(extra="Restart_Run: .true.\nFix_Number_Eigenvalues: .true.").replace(
                    "States_Num: 4", "States_Num: 6"
                ),
                encoding="utf-8",
            )
            code, error = _run_cli(path)
            self.assertEqual(code, 1)
            self.assertIn("PARSEC error 404", error + (directory / "parsec.out").read_text())

    def test_accelerated_cli_runs_it(self) -> None:
        # The accelerated eigensolvers keep no buffer and the symmetry sectors
        # their counts (acceleration/tests/test_geigen_parity.py).
        from parsec_python.acceleration.cli import _reference_only_features

        translation = _parse(H2, extra="Restart_Run: .true.\nFix_Number_Eigenvalues: .true.")
        features = _reference_only_features(translation, accelerated_backend_selected=False)
        self.assertFalse(any("Fix_Number_Eigenvalues" in item for item in features))


# ----------------------------------------------------------------------------
# 2. Dynamic_Diag_Tol in every driver
# ----------------------------------------------------------------------------


class DynamicToleranceTests(unittest.TestCase):
    def test_parser_accepts_kpoints_and_spinors_and_drops_it_for_restarts(self) -> None:
        translation = _parse(CELL, iterations=40, extra="Dynamic_Diag_Tol: .true.")
        self.assertTrue(translation.problem.eigensolver.dynamic_tolerance)
        spinor = _parse(AU_CELL, iterations=40, extra="Dynamic_Diag_Tol: .true.")
        self.assertTrue(spinor.problem.eigensolver.dynamic_tolerance)
        restart = _parse(H2, extra="Dynamic_Diag_Tol: .true.\nRestart_Run: .true.")
        self.assertFalse(restart.problem.eigensolver.dynamic_tolerance)
        self.assertTrue(restart.dynamic_tolerance_off_for_restart)
        self.assertTrue(any("Dynamic_Diag_Tol is not used with Restart_Run" in w for w in restart.warnings))

    def test_kpoint_solver_passes_the_tolerance(self) -> None:
        # Both k-point solvers: the serial reference definition and the worker
        # pool of the acceleration package (on the CPU).
        from parsec_python.SCF.kpoint_solver import KPointJob, SerialKPointSolver
        from parsec_python.acceleration.SCF.kpoint_parallel import PooledKPointSolver

        for make in (
            lambda job: SerialKPointSolver(job),
            lambda job: PooledKPointSolver(job, workers=1, backend="cpu"),
        ):
            with self.subTest(solver=make):
                self._check_tolerance_passed(KPointJob, make)

    def _check_tolerance_passed(self, KPointJob, make) -> None:
        seen = []

        def settings(degree, tolerance=None):
            from parsec_python.Eigensolvers import EigvalSettings

            seen.append((degree, tolerance))
            return EigvalSettings()

        matrix = np.diag(np.arange(1.0, 13.0))

        class Operator:
            def as_linear_operator(self):
                from scipy.sparse.linalg import aslinearoperator

                return aslinearoperator(matrix)

        job = KPointJob(1, 2, settings, lambda task, potentials: Operator(), lambda *a, **k: {})
        with make(job) as solver:
            solver.solve({}, 15)
            solver.solve({}, 14, 0.05)
        self.assertEqual(seen, [(15, None), (14, 0.05)])

    def test_kpoint_loop_records_the_tolerance_and_keeps_the_energy(self) -> None:
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.Grid.kpoints import kpoint_sampling

        energies = {}
        for flag in (False, True):
            translation = _parse_kept(
                self, CELL, iterations=40, extra="Dynamic_Diag_Tol: .true." if flag else ""
            )
            system = prepare_periodic_single_point(translation.problem)
            k_points, weights = kpoint_sampling(translation.problem)
            records = []
            pin_kpoints_on_the_cpu(self)
            result = run_scf_kpoints(system, k_points, weights, callback=records.append)
            self.assertTrue(result.converged)
            energies[flag] = result.energies.total
            tolerances = [step.next_eigensolver_tolerance for step in records]
            if flag:
                self.assertTrue(all(value is not None for value in tolerances))
                self.assertLessEqual(tolerances[0], 0.1)
                self.assertTrue(all(b <= a for a, b in zip(tolerances, tolerances[1:])))
                self.assertGreaterEqual(min(tolerances), 1.0e-7)
            else:
                self.assertTrue(all(value is None for value in tolerances))
        self.assertAlmostEqual(energies[True], energies[False], places=6)

    def test_spin_and_spinor_reports_print_the_lines(self) -> None:
        from parsec_python.Output.parsec_output import dynamic_tolerance_lines
        from parsec_python.models import SelfConsistentSOCIteration, SpinPolarizedSCFIteration

        self.assertEqual(dynamic_tolerance_lines(object()), [])
        for record in (
            SpinPolarizedSCFIteration(1, 0.1, 0.2, None, next_eigensolver_tolerance=0.012345),
            SelfConsistentSOCIteration(1, 0.1, 0.2, None, next_eigensolver_tolerance=0.012345),
        ):
            self.assertEqual(
                dynamic_tolerance_lines(record),
                ["Dynamic tolerance is used in eigensolver", "", "New tolerance:    0.01235", ""],
            )

    def test_spin_orbit_settings_take_the_tolerance(self) -> None:
        from parsec_python.SCF.kpoints_soc import _eigval_settings
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        translation = _parse_kept(
            self, AU_CELL, iterations=1, extra="Dynamic_Diag_Tol: .true."
        ).problem
        from dataclasses import replace

        problem = replace(
            translation, eigensolver=replace(translation.eigensolver, method="chebdav")
        )
        system = prepare_periodic_single_point(problem)
        settings = _eigval_settings(system, 15, 0.03)
        self.assertEqual(settings.chebdav.convergence_tolerance, 0.03)
        self.assertEqual(
            _eigval_settings(system, 15).chebdav.convergence_tolerance, problem.eigensolver.tolerance
        )

    def test_spin_orbit_kpoint_loop_hands_the_tolerance_to_its_solver(self) -> None:
        # The spinor k-point job took no tolerance (TypeError at the first solve of
        # run_self_consistent_soc_kpoints with Dynamic_Diag_Tol; found by the merge
        # of the device k-point solver).
        from dataclasses import replace

        from parsec_python.SCF.kpoints_soc import run_self_consistent_soc_kpoints
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        translation = _parse_kept(
            self, AU_CELL, iterations=3, extra="Dynamic_Diag_Tol: .true."
        ).problem
        problem = replace(
            translation, eigensolver=replace(translation.eigensolver, method="chebdav")
        )
        system = prepare_periodic_single_point(problem)
        records = []
        pin_kpoints_on_the_cpu(self)
        run_self_consistent_soc_kpoints(
            system, np.zeros((1, 3)), np.ones(1), callback=records.append
        )
        tolerances = [step.next_eigensolver_tolerance for step in records]
        self.assertEqual(len(tolerances), 3)
        self.assertTrue(all(value is not None and value <= 0.1 for value in tolerances))


# ----------------------------------------------------------------------------
# 3. Save_Intermediate_* in the extended drivers
# ----------------------------------------------------------------------------


class IntermediateFilesTests(unittest.TestCase):
    def test_kpoint_run_writes_every_iteration(self) -> None:
        from parsec_python.Output import output_files
        from parsec_python.Output.parsec_dat import read_parsec_dat

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = _write(
                directory,
                CELL,
                iterations=3,
                extra=(
                    "Save_Intermediate_Charge_Density: .true.\n"
                    "Save_Intermediate_Eigenvalues: .true.\nOutput_All_States: .true."
                ),
            )
            written = []
            original = output_files.RunOutputFiles._write_parsec_dat

            def counting(self, result, grid, problem, *, report):
                written.append(report)
                return original(self, result, grid, problem, report=report)

            output_files.RunOutputFiles._write_parsec_dat = counting
            try:
                code, error = _run_cli(path)
            finally:
                output_files.RunOutputFiles._write_parsec_dat = original
            self.assertEqual(code, 3, error)
            self.assertEqual(written, [False, False, False, True])
            lines = (directory / "eigen.dat").read_text().splitlines()
            # Header, then 3 iterations x 2 k-points x 4 states (format 12).
            self.assertEqual(len(lines), 1 + 3 * 2 * 4)
            self.assertIn("k-point", lines[0])
            data = read_parsec_dat(directory / "parsec.dat", wavefunctions=False)
            self.assertEqual(data.periodic.k_points.shape, (2, 3))

    def test_spinor_iteration_fields_write_a_spin_orbit_file(self) -> None:
        from parsec_python.Output.output_files import RunOutputFiles
        from parsec_python.Output.parsec_dat import read_parsec_dat
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.SCF.kpoints_soc import run_self_consistent_soc_kpoints

        translation = _parse_kept(
            self, AU_CELL, iterations=2, extra="Save_Intermediate_Charge_Density: .true."
        )
        system = prepare_periodic_single_point(translation.problem)
        with tempfile.TemporaryDirectory() as folder:
            files = RunOutputFiles(translation.output_files, folder, write=lambda line: None, kpoint_count=1)
            files.plan_wavefunctions(system.grid.size, 14, 2, complex_values=True)
            files.start()
            states = []

            def callback(state):
                states.append(state.iteration)
                files.iteration_fields(state, system.grid, translation.problem)

            pin_kpoints_on_the_cpu(self)
            run_self_consistent_soc_kpoints(
                system, np.zeros((1, 3)), np.ones(1), iteration_state_callback=callback
            )
            self.assertEqual(states, [1, 2])
            data = read_parsec_dat(Path(folder) / "parsec.dat", wavefunctions=False)
            self.assertEqual(data.wavefunction_kind, 2)
            self.assertEqual(len(data.potentials), 2)

    def test_parser_accepts_every_driver(self) -> None:
        for template, fields in (
            (CELL, {"iterations": 3}),
            (AU_CELL, {"iterations": 3}),
        ):
            translation = _parse(
                template,
                extra=(
                    "Save_Intermediate_Charge_Density: .true.\nOutput_Eigenvalues: .true.\n"
                    "Save_Intermediate_Eigenvalues: .true."
                ),
                **fields,
            )
            self.assertTrue(translation.output_files.intermediate_density)
            self.assertTrue(translation.output_files.intermediate_eigenvalues)


# ----------------------------------------------------------------------------
# 4. nscf_kpoints with spin-orbit spinors
# ----------------------------------------------------------------------------


class SpinOrbitNSCFTests(unittest.TestCase):
    def test_spinor_fermi_level_counts_single_electrons(self) -> None:
        from parsec_python.SCF.nscf import fermi_level_from_levels

        levels = [[np.array([-1.0, -0.5, 0.5, 1.0])]]
        scalar = fermi_level_from_levels(levels, [1.0], 2.0, 300.0)
        spinor = fermi_level_from_levels(levels, [1.0], 2.0, 300.0, spinors=True)
        self.assertLess(scalar, 0.0)  # one pair fills the lowest level
        self.assertTrue(-0.5 < spinor < 0.5)  # two spinor states hold two electrons

    def test_parser_takes_spin_orbit_and_refuses_noncollinear(self) -> None:
        nscf = "Restart_Run: .true.\nbegin nscf_kpoints\n0 0 0 1\n0.1 0 0 1\nend nscf_kpoints\n"
        translation = _parse(AU_CELL, iterations=40, extra=nscf)
        self.assertIsNotNone(translation.nscf)
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "non-collinear"):
            _parse(AU_CELL, iterations=40, extra=nscf + "Non_Collinear_magnetism: .true.\n")

    def test_nscf_at_the_scf_kpoint_gives_the_scf_levels(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = _write(directory, AU_CELL, iterations=60, extra="")
            code, error = _run_cli(path)
            self.assertEqual(code, 0, error)
            with np.load(directory / "parsec_python_results.npz") as archive:
                scf_levels = np.sort(np.asarray(archive["eigenvalues_ry"]))
            path.write_text(
                AU_CELL.format(
                    iterations=60,
                    extra=(
                        "Restart_Run: .true.\nnscf_states_num: 10\n"
                        "begin nscf_kpoints\n0 0 0 1\n0.25 0 0 1\nend nscf_kpoints\n"
                    ),
                ),
                encoding="utf-8",
            )
            code, error = _run_cli(path)
            self.assertEqual(code, 0, error)
            with np.load(directory / "parsec_python_nscf.npz") as archive:
                values = np.asarray(archive["nscf_eigenvalues_ry"])
            self.assertEqual(values.shape, (1, 2, 10))
            # Gamma is the SCF's k-point: the same spinor levels.
            np.testing.assert_allclose(values[0, 0], scf_levels[:10], atol=2e-4)
            log = (directory / "parsec.out").read_text()
            self.assertIn(" Spin-Orbit: each state has a magnetic moment!", log)
            self.assertIn(
                "up State   Eigenv.[Ry]      Eigenv.[eV]    Occup.     Repr.   Spin_z [hbar]", log
            )

    def test_report_formats(self) -> None:
        from parsec_python.Output.post_processing_output import nscf_report_lines
        from parsec_python.SCF.nscf import NSCFResult

        result = NSCFResult(
            channels=("all",),
            k_points=np.zeros((1, 3)),
            weights=np.ones(1),
            eigenvalues=np.array([[[-0.449298, -0.449295]]]),
            occupations=np.array([[[1.0, 1.0]]]),
            fermi_level=0.0,
            temperature_kelvin=300.0,
            residual_norms=np.zeros((1, 1)),
            attempts=np.ones((1, 1), dtype=int),
            electrons=(2.0,),
            spin_z=np.array([[0.0116, -0.0116]]),
        )
        lines = nscf_report_lines(result, np.eye(3) * 8.0, matvec_budget=98328, diagonalization_seconds=4.5)
        self.assertIn(" Net magnetic moment (in Bohr magnetons):    0.00", lines)
        # Format 65: (i5,3x,f14.6,3x,f14.6,1x,f9.4,3x,i6,3x,f9.4).
        value = -0.449298
        self.assertIn(
            f"{1:5d}   {value:14.6f}   {value * 13.6058:14.6f} {1.0:9.4f}   {1:6d}   {0.0116:9.4f}",
            lines,
        )


# ----------------------------------------------------------------------------
# 5. Symmetry printouts
# ----------------------------------------------------------------------------

_H2O_ATOMS = [
    Atom(symbol="H", position=(0.0, -0.519674891, 1.795240534)),
    Atom(symbol="H", position=(0.0, -0.519674891, -1.795240534)),
    Atom(symbol="O", position=(0.0, 0.519674891, 0.0)),
]

# Fortran PARSEC (WSL binary), H2O sphere 10.4 bohr, Output_Level 2.
_H2O_LEVEL2 = """
 Symmetry Properties:
 --------------------

 The crystallographic system is cubic with operations:
       1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
      17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
      33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48


 The space group of the crystal is symmorphic
 (this usually means that symmetry operations do not
 involve rotations and translations simultaneously)

 Operation number:
       1  3 26 28

   Multiplication table
    x   1   3  26  28
    1   1   3  26  28
    3   3   1  28  26
   26  26  28   1   3
   28  28  26   3   1


    Rotation matrices (r-lattice) and fractional translations (r-lattice)

    1     1  0  0     0  1  0     0  0  1     -0.00000 -0.00000 -0.00000   E
    2    -1  0  0     0  1  0     0  0 -1     -0.00000 -0.00000 -0.00000   C2
    3    -1  0  0     0  1  0     0  0  1     -0.00000 -0.00000 -0.00000   IC2
    4     1  0  0     0  1  0     0  0 -1     -0.00000 -0.00000 -0.00000   IC2
 Checking in r-space
 Multiplication table

  1 2 3 4
  2 1 4 3
  3 4 1 2
  4 3 2 1

  Found Abelian subgroup C_2v  with            4  symmetry operations

  Character table : character = exp(i*Pi*x/3)
--------------------------------------
 Representation |    E   C2  IC2  IC2
   A1      1   |     0    0    0    0
   A2      2   |     0    0    3    3
   B1      3   |     0    3    0    3
   B2      4   |     0    3    3    0
--------------------------------------
 Operation number:   1    2    3    4
"""


class SymmetryPrintoutTests(unittest.TestCase):
    def test_h2o_block_equals_fortran(self) -> None:
        from parsec_python.Grid.symmetry_analysis import report_space_group
        from parsec_python.Output.kpoint_output import space_group_lines

        group = report_space_group(np.eye(3) * 4 * 10.4, _H2O_ATOMS)
        lines = space_group_lines(group, info=0, norot=0, output_level=2)
        expected = _H2O_LEVEL2.splitlines()
        self.assertEqual([line.rstrip() for line in lines[: len(expected)]], [line.rstrip() for line in expected])
        self.assertEqual(lines[len(expected):], [""])
        level1 = space_group_lines(group, info=0, norot=0, output_level=1)
        self.assertNotIn("   Multiplication table", level1)
        self.assertNotIn(" Checking in r-space", level1)
        self.assertIn("  Found Abelian subgroup C_2v  with            4  symmetry operations", level1)

    def test_fast_search_equals_the_port_of_atftmt(self) -> None:
        from parsec_python.Grid.space_group import _crystal_operations, space_group_with_search
        from parsec_python.Grid.symmetry_analysis import report_space_group

        a = 0.6293 / 0.529177
        methane = [Atom("H", (a, a, a)), Atom("H", (a, -a, -a)), Atom("H", (-a, a, -a)), Atom("H", (-a, -a, a)), Atom("C", (0.0, 0.0, 0.0))]
        fractions = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0], [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]])
        diamond = [Atom("Si", tuple(row * 10.26)) for row in fractions]
        fcc = np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) * 10.26
        silicon = [Atom("Si", tuple(row)) for row in np.array([[0, 0, 0], [.25, .25, .25]]) @ fcc]
        hcp = np.array([[3.21, 0, 0], [-1.605, 3.21 * np.sqrt(3) / 2, 0], [0, 0, 5.21]]) / 0.529177
        magnesium = [Atom("Mg", tuple(row)) for row in np.array([[1 / 3, 2 / 3, .25], [2 / 3, 1 / 3, .75]]) @ hcp]
        doubled = [Atom("Al", (0.0, 0.0, 0.0)), Atom("Al", (6.0, 0.0, 0.0))]
        for lattice, atoms in (
            (np.eye(3) * 36.0, methane),
            (np.eye(3) * 41.6, _H2O_ATOMS),
            (np.eye(3) * 10.26, diamond),
            (fcc, silicon),
            (hcp, magnesium),
            (np.diag([12.0, 6.0, 6.0]), doubled),
        ):
            # find_space_group (which the report takes) searches with the neighbour tree; the dense port of
            # atftmt is the reference
            self.assertEqual(
                report_space_group(lattice, atoms), space_group_with_search(lattice, atoms, _crystal_operations)
            )

    def test_symchk_and_abelian(self) -> None:
        from parsec_python.Grid.symmetry_analysis import (
            abelian_subgroup,
            report_space_group,
            symmetry_check,
        )

        a = 1.189205
        methane = report_space_group(
            np.eye(3) * 36.0,
            [Atom("H", (a, a, a)), Atom("H", (a, -a, -a)), Atom("H", (-a, a, -a)), Atom("H", (-a, -a, a)), Atom("C", (0.0, 0.0, 0.0))],
        )
        check = symmetry_check(methane)
        self.assertEqual(check.error, 0)
        subgroup = abelian_subgroup(methane, check.table, 0)
        self.assertEqual(subgroup.name, "D_2  ")
        self.assertEqual(subgroup.operations, (1, 2, 3, 4))
        self.assertEqual(subgroup.character_exponents()[1], (0, 0, 3, 3))
        fractions = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0], [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]])
        diamond = report_space_group(np.eye(3) * 10.26, [Atom("Si", tuple(row * 10.26)) for row in fractions])
        # PARSEC's translation test fails for the diamond operations with 1/4.
        check = symmetry_check(diamond)
        self.assertEqual(check.error, 14)
        self.assertIn((13, 2, 14, 2, 0.5), check.failures)

    def test_search_mode_follows_parsec(self) -> None:
        from parsec_python.Grid.symmetry_analysis import abelian_rotation_flag, symmetry_search_mode

        plain = dict(
            ignore_symmetry=False, kpoints=False, grid_shift=(0.5, 0.5, 0.5), periodicity=0,
            spinors=False, bands=False, nscf=False, molecular_dynamics=False,
            polarizability=False, manual_movement=False, non_spherical_cluster=False,
        )
        self.assertEqual(symmetry_search_mode(**plain).info, 0)
        self.assertEqual(symmetry_search_mode(**{**plain, "kpoints": True}).info, 1)
        self.assertEqual(symmetry_search_mode(**{**plain, "spinors": True}).info, 1)
        self.assertEqual(symmetry_search_mode(**{**plain, "grid_shift": (0.0, 0.5, 0.5)}).info, 1)
        self.assertEqual(symmetry_search_mode(**{**plain, "periodicity": 2}).info, 2)
        self.assertEqual(symmetry_search_mode(**{**plain, "kpoints": True, "molecular_dynamics": True}).info, 2)
        mode = symmetry_search_mode(**{**plain, "non_spherical_cluster": True})
        self.assertEqual((mode.info, mode.domain_warning), (2, True))
        self.assertTrue(symmetry_search_mode(**{**plain, "ignore_symmetry": True}).suppressed)
        self.assertEqual(abelian_rotation_flag(np.eye(3), 0), 0)
        self.assertEqual(abelian_rotation_flag(np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]), 0), 1)
        self.assertEqual(abelian_rotation_flag(np.eye(3), 1), -1)

    def test_cluster_report_carries_the_block(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = _write(directory, H2.replace("Ignore_Symmetry: .true.\n", ""), extra="Max_Iter: 1")
            text = path.read_text().replace("Max_Iter: 40\n", "")
            path.write_text(text)
            _run_cli(path)
            log = (directory / "parsec.out").read_text()
            self.assertIn(" Symmetry Properties:", log)
            self.assertIn("  Found Abelian subgroup D_2h  with            8  symmetry operations", log)
            # 6. the ion-ion force table of forceion.f90 (Output_Level 1).
            self.assertIn(" Forces from ion-ion interaction:", log)


if __name__ == "__main__":
    unittest.main()
