"""Eigensolver features of the Fortran-parity round on the accelerated driver.

``Dynamic_Diag_Tol`` (the tolerance of every SCF iteration reaches the device
CHEBDAV first solves and ARPACK), ``Fix_Number_Eigenvalues`` (no buffer;
symmetry sectors keep the counts of their first solve), ``Eigensolver: none``
(PARSEC's TEST mode after the accelerated setup) and ``Eigensolver: arpack``
on the device (:mod:`parsec_python.acceleration.Eigensolvers.arpack`).
DIAGLA stays on the reference solver.  The numerical tests need CuPy and a
CUDA device; the routing and report tests do not.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from parsec_python.Eigensolvers import EigvalSettings, SubspaceSettings
from parsec_python.Eigensolvers.arpack import ArpackSettings, run_arpack
from parsec_python.Eigensolvers.iterative import IterativeDiagonalizationError
from parsec_python.Input import parse_parsec_input
from parsec_python.acceleration.backends.cupy import cupy_available

GPU_AVAILABLE = cupy_available()
DATA = Path(__file__).resolve().parents[2] / "tests" / "data"

H2 = """\
Boundary_Conditions: cluster
Boundary_Sphere_Radius: 5.0 bohr
Grid_Spacing: 0.5 bohr
Coordinate_Unit: Cartesian_Bohr
Ignore_Symmetry: {ignore}
Hartree_Boundary_Tolerance: off
Expansion_Order: 4
States_Num: 4
Fermi_Temp: 500.0
Max_Iter: 40
Convergence_Criterion: 1.0e-7
Diag_Tolerance: 1.0e-8
Eigensolver: {solver}
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


def _write(directory: Path, *, solver="chebff", ignore=".false.", extra="") -> Path:
    shutil.copy(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
    path = directory / "parsec.in"
    path.write_text(H2.format(solver=solver, ignore=ignore, extra=extra), encoding="utf-8")
    return path


def _parse(**fields):
    with tempfile.TemporaryDirectory() as folder:
        return parse_parsec_input(_write(Path(folder), **fields))


def _run(path: Path, *extra: str, reference: bool = False) -> tuple[int, str]:
    if reference:
        from parsec_python.cli import main
    else:
        from parsec_python.acceleration.cli import main
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as error:
        code = main([str(path), "--quiet", *extra])
    return code, error.getvalue()


def _archive(directory: Path) -> dict:
    with np.load(directory / "parsec_python_results.npz", allow_pickle=True) as archive:
        values = {key: archive[key] for key in archive.files}
    details = dict(zip(values.get("backend_detail_keys", ()), values.get("backend_detail_values", ())))
    values["details"] = details
    return values


def _chain(dimension: int, seed: int = 0) -> sp.csr_matrix:
    """A symmetric second-difference operator plus a random diagonal."""

    generator = np.random.default_rng(seed)
    main = 2.0 + generator.uniform(-0.5, 0.5, dimension)
    off = -np.ones(dimension - 1)
    return sp.diags([off, main, off], [-1, 0, 1], format="csr") * 10.0


class _Operator:
    """A device operator of a host matrix: what the ARPACK loop applies."""

    def __init__(self, matrix):
        import cupyx.scipy.sparse as device_sparse

        self.matrix = device_sparse.csr_matrix(matrix)
        self.shape = matrix.shape

    def apply(self, vectors):
        return self.matrix @ vectors


# ----------------------------------------------------------------------------
# Routing and report (no device needed)
# ----------------------------------------------------------------------------


class RoutingTests(unittest.TestCase):
    def features(self, translation, *, device: bool) -> list[str]:
        from parsec_python.acceleration.cli import _reference_only_features

        return _reference_only_features(
            translation,
            accelerated_backend_selected=False,
            device_eigensolver_selected=device,
        )

    def test_controls_and_test_mode_run_on_the_accelerated_path(self) -> None:
        for fields in (
            dict(solver="chebdav", extra="Dynamic_Diag_Tol: .true."),
            dict(solver="chebff", extra="Restart_Run: .true.\nFix_Number_Eigenvalues: .true."),
            dict(solver="none"),
        ):
            with self.subTest(fields=fields):
                self.assertEqual(self.features(_parse(**fields), device=False), [])

    def test_arpack_runs_on_the_device_only_with_the_cupy_eigensolvers(self) -> None:
        translation = _parse(solver="arpack")
        self.assertEqual(self.features(translation, device=True), [])
        self.assertTrue(
            any("ARPACK" in item for item in self.features(translation, device=False))
        )

    def test_arpack_of_a_periodic_cell_stays_on_the_reference_solver(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            shutil.copy(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
            text = H2.format(solver="arpack", ignore=".false.", extra="").replace(
                "Boundary_Conditions: cluster\nBoundary_Sphere_Radius: 5.0 bohr\n",
                "Periodic_System: .true.\nBoundary_Conditions: bulk\n"
                "begin Cell_Shape\n  6.0 6.0 6.0\nend Cell_Shape\n",
            )
            (directory / "parsec.in").write_text(text, encoding="utf-8")
            translation = parse_parsec_input(directory / "parsec.in")
        self.assertIsNotNone(translation.problem.periodic_cell)
        self.assertTrue(
            any("ARPACK" in item for item in self.features(translation, device=True))
        )

    def test_diagla_stays_on_the_reference_solver(self) -> None:
        translation = _parse(solver="diagla")
        for device in (False, True):
            self.assertTrue(
                any("DIAGLA" in item for item in self.features(translation, device=device))
            )

    def test_count_lines_list_one_value_per_representation(self) -> None:
        from parsec_python.Output.parsec_output import iterative_eigensolver_count_lines
        from parsec_python.SCF.single_point import iteration_eigensolver_fields

        lines, total = iterative_eigensolver_count_lines([(14, 592)], 0, False)
        self.assertEqual(
            lines,
            [
                " Number of converged eigen-pairs per representation:          14",
                " Number of matrix-vector multiplications:          592",
                " Cumulated number of matrix-vector mult. =          592",
            ],
        )
        fields = iteration_eigensolver_fields((((8, 6), (397, 337)),), None)
        self.assertEqual(fields["eigensolver_counts"], (((8, 6), (397, 337)),))
        lines, total = iterative_eigensolver_count_lines(
            fields["eigensolver_counts"], total, False
        )
        self.assertEqual(
            lines[0],
            " Number of converged eigen-pairs per representation:           8           6",
        )
        self.assertEqual(
            lines[1], " Number of matrix-vector multiplications:          397         337"
        )
        self.assertEqual(lines[2], f" Cumulated number of matrix-vector mult. = {592 + 734:12d}")


# ----------------------------------------------------------------------------
# The device ARPACK loop
# ----------------------------------------------------------------------------


@unittest.skipUnless(GPU_AVAILABLE, "CuPy with a CUDA device is required")
class DeviceArpackTests(unittest.TestCase):
    def test_pairs_and_start_vector_of_the_reference(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.Eigensolvers.arpack import run_device_arpack

        matrix = _chain(300)
        settings = ArpackSettings(tolerance=1.0e-10)
        device = run_device_arpack(_Operator(matrix), 12, settings=settings, matvec_budget=10**6)
        reference = run_arpack(matrix, 12, settings=settings, matvec_budget=10**6)
        exact = np.linalg.eigvalsh(matrix.toarray())[:12]
        values = cp.asnumpy(device.eigenvalues)
        vectors = cp.asnumpy(device.vectors)
        np.testing.assert_allclose(values, exact, atol=1.0e-9)
        np.testing.assert_allclose(values, reference.eigenvalues, atol=1.0e-9)
        np.testing.assert_allclose(vectors.T @ vectors, np.eye(12), atol=1.0e-10)
        residuals = np.linalg.norm(matrix @ vectors - vectors * values, axis=0)
        self.assertLess(float(residuals.max()), 1.0e-6)
        self.assertEqual(device.converged_states, 12)
        self.assertTrue(device.vectors.flags.f_contiguous)
        # Both drew one dlarnv start vector of 300 values from 1, 3, 5, 7.
        self.assertEqual(len(device.calls), 1)
        self.assertEqual(device.start_seed, reference.start_seed)

    def test_budget_is_error_402(self) -> None:
        from parsec_python.acceleration.Eigensolvers.arpack import run_device_arpack

        with self.assertRaises(IterativeDiagonalizationError) as caught:
            run_device_arpack(
                _Operator(_chain(300)), 12, settings=ArpackSettings(tolerance=1.0e-10),
                matvec_budget=20,
            )
        self.assertEqual(caught.exception.ierr, 402)

    def test_a_tiny_sector_is_solved_exactly(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.Eigensolvers.arpack import run_device_arpack

        matrix = _chain(8, seed=3)
        result = run_device_arpack(_Operator(matrix), 8, matvec_budget=10**6)
        np.testing.assert_allclose(
            cp.asnumpy(result.eigenvalues), np.linalg.eigvalsh(matrix.toarray()), atol=1.0e-12
        )

    def test_solver_runs_arpack_every_call_or_once_with_filtering(self) -> None:
        from parsec_python.acceleration.backends.cupy import CuPyHamiltonian
        from parsec_python.acceleration.Eigensolvers.eigval import CuPyEigvalSolver

        matrix = _chain(400, seed=1)
        for filtering, paths in ((None, ["arpack"] * 3), (True, ["arpack", "subspace", "subspace"])):
            with self.subTest(filtering=filtering):
                settings = EigvalSettings(
                    safety_buffer=6,
                    initial_method="arpack",
                    arpack=ArpackSettings(tolerance=1.0e-10),
                    chebyshev_filtering=filtering,
                    subspace=SubspaceSettings(polynomial_degree=12, block_size=3),
                )
                solver = CuPyEigvalSolver(
                    CuPyHamiltonian(matrix, np.zeros(400)), settings=settings
                )
                seen, seeds = [], []
                for _ in range(3):
                    result = solver.solve(6)
                    seen.append(result.solver_path)
                    seeds.append(result.state.start_seed)
                    np.testing.assert_allclose(
                        result.eigenvalues, np.linalg.eigvalsh(matrix.toarray())[:6], atol=1.0e-8
                    )
                self.assertEqual(seen, paths)
                if filtering is None:
                    # Every call draws a new start vector (info = 0).
                    self.assertEqual(len(set(seeds)), 3)
                    self.assertEqual(result.iterative_counts[0], 12)
                else:
                    self.assertIsNone(result.iterative_counts)


# ----------------------------------------------------------------------------
# Complete calculations on the accelerated CLI
# ----------------------------------------------------------------------------


@unittest.skipUnless(GPU_AVAILABLE, "CuPy with a CUDA device is required")
class AcceleratedCalculationTests(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name)

    def test_test_mode_stops_after_the_accelerated_setup(self) -> None:
        path = _write(self.directory, solver="none")
        code, error = _run(path)
        self.assertEqual(code, 0, error)
        self.assertNotIn("reference parsec_python.cli solver", error)
        log = (path.parent / "parsec.out").read_text()
        self.assertIn(" Eigensolver is 'none'. Preparing for Special TEST MODE", log)
        self.assertIn(" TEST MODE COMPLETE", log)
        self.assertIn("orbital_symmetry", log)
        self.assertNotIn("SRE", log)
        self.assertFalse((path.parent / "parsec_python_results.npz").exists())

    def test_arpack_on_symmetry_sectors_matches_the_reference(self) -> None:
        reference_directory = self.directory / "reference"
        reference_directory.mkdir()
        code, error = _run(_write(reference_directory, solver="arpack"), reference=True)
        self.assertEqual(code, 0, error)
        path = _write(self.directory, solver="arpack")
        code, error = _run(path)
        self.assertEqual(code, 0, error)
        self.assertNotIn("reference parsec_python.cli solver", error)
        accelerated, reference = _archive(self.directory), _archive(reference_directory)
        self.assertEqual(str(accelerated["backend_selected"]), "cupy")
        self.assertLess(
            abs(float(accelerated["energy_total_ry"]) - float(reference["energy_total_ry"])), 1.0e-7
        )
        np.testing.assert_allclose(
            accelerated["eigenvalues_ry"], reference["eigenvalues_ry"], atol=1.0e-6
        )
        log = (path.parent / "parsec.out").read_text()
        # H2 on the z axis: the eight sectors of D_2h, each counted.
        counts = [line for line in log.splitlines() if "converged eigen-pairs" in line]
        self.assertTrue(counts)
        self.assertEqual(len(counts[0].split(":")[1].split()), 8)

    def test_dynamic_tolerance_reaches_the_device_chebdav(self) -> None:
        from parsec_python.acceleration import driver
        import parsec_python.acceleration.Eigensolvers.eigval as device_policy

        translation = parse_parsec_input(
            _write(self.directory, solver="chebdav", extra="Dynamic_Diag_Tol: .true.")
        )
        system = driver.prepare_single_point(translation.problem, backend="cupy", symmetry="auto")
        tolerances = []
        original = device_policy.run_chebdav

        def recorded(operator, wanted, *, settings, **options):
            tolerances.append(settings.convergence_tolerance)
            return original(operator, wanted, settings=settings, **options)

        with patch.object(device_policy, "run_chebdav", recorded):
            result = driver.run_scf(system)
        self.assertTrue(result.converged)
        # One CHEBDAV first solve per sector, at the 0.1 of the first iteration.
        self.assertTrue(tolerances)
        self.assertEqual(set(tolerances), {0.1})
        history = [step.next_eigensolver_tolerance for step in result.history]
        self.assertTrue(all(value is not None for value in history))
        self.assertLess(history[-1], 0.1)
        self.assertGreaterEqual(min(history), 1.0e-8)
        self.assertEqual(history, sorted(history, reverse=True))

    def test_fixed_number_of_eigenvalues_after_a_restart(self) -> None:
        path = _write(self.directory)
        code, error = _run(path)
        self.assertEqual(code, 0, error)
        energy = float(_archive(self.directory)["energy_total_ry"])
        path.write_text(
            H2.format(
                solver="chebff",
                ignore=".false.",
                extra="Restart_Run: .true.\nFix_Number_Eigenvalues: .true.",
            ),
            encoding="utf-8",
        )
        code, error = _run(path)
        self.assertEqual(code, 0, error)
        self.assertNotIn("reference parsec_python.cli solver", error)
        archive = _archive(self.directory)
        self.assertLess(abs(float(archive["energy_total_ry"]) - energy), 1.0e-6)
        counts = [int(value) for value in archive["details"]["orbital_sector_final_state_counts"].split()]
        share = np.bincount(np.asarray(archive["representations"]) - 1, minlength=len(counts))
        self.assertEqual(counts, [max(1, int(value)) for value in share])
        log = (path.parent / "parsec.out").read_text()
        self.assertIn(" Fixing the number of eigenvalues per representation", log)
        # Six states from a file of four: PARSEC's error 404.
        path.write_text(
            H2.format(
                solver="chebff",
                ignore=".false.",
                extra="Restart_Run: .true.\nFix_Number_Eigenvalues: .true.",
            ).replace("States_Num: 4", "States_Num: 6"),
            encoding="utf-8",
        )
        code, error = _run(path)
        self.assertEqual(code, 1)
        self.assertIn("404", error + (path.parent / "parsec.out").read_text())

    def test_restart_after_a_centring_first_run_reaches_the_fortran_energies(self) -> None:
        # Restart_Run keeps the input coordinates, as PARSEC does
        # (tests/unit/test_restart_coordinates.py has the reference CLI).
        from parsec_python.acceleration.cli import main
        from parsec_python.tests.unit.test_restart_coordinates import (
            check_against_fortran,
            first_and_restart,
        )

        check_against_fortran(self, first_and_restart(self, main))


if __name__ == "__main__":
    unittest.main()
