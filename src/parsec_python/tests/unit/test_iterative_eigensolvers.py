"""ARPACK, DIAGLA, the eigval dispatch of both and the new eigensolver controls."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from parsec_python.Eigensolvers import (
    ArpackSettings,
    DiaglaSettings,
    EigvalSettings,
    IterativeDiagonalizationError,
    dynamic_diagonalization_tolerance,
    fdo5scl_preconditioner,
    matvec_budget,
    run_arpack,
    run_diagla,
    solve_eigval,
)
from parsec_python.Eigensolvers.arpack import ARPACK_START_SEED, arpack_start_vector
from parsec_python.Eigensolvers.diagla import diagla_diag
from parsec_python.Eigensolvers.lapack_random import LapackRandom
from parsec_python.Grid import build_cluster_grid, nearest_neighbor_rows
from parsec_python.Input import parse_parsec_input
from parsec_python.Input.parsec_input import UnsupportedParsecOptionError
from parsec_python.models import (
    Atom,
    EigensolverSettings,
    GridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.SCF import prepare_single_point, run_scf

DATA = Path(__file__).parents[1] / "data"
H2_INPUT = DATA / "H2_parsec.in"


def _box_grid(n: int):
    """A cubic n**3 grid as a stand-in for a periodic grid without wrap."""

    axis = np.arange(n)
    mesh = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.column_stack([part.reshape(-1) for part in mesh])
    lookup = np.arange(n**3).reshape(n, n, n)

    class Grid:
        integer_coordinates = points

        @staticmethod
        def rows_for_integer_coordinates(query):
            inside = np.all((query >= 0) & (query < n), axis=1)
            rows = np.full(len(query), -1, dtype=np.int64)
            rows[inside] = lookup[query[inside, 0], query[inside, 1], query[inside, 2]]
            return rows

    return Grid()


def _laplacian_matrix(n: int, *, seed: int = 0, complex_part: float = 0.0):
    """-nabla^2 on an n**3 grid plus a random diagonal (a model Hamiltonian)."""

    identity = sp.identity(n)
    second = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    laplacian = (
        sp.kron(sp.kron(second, identity), identity)
        + sp.kron(sp.kron(identity, second), identity)
        + sp.kron(sp.kron(identity, identity), second)
    )
    rng = np.random.default_rng(seed)
    matrix = laplacian + sp.diags(rng.uniform(-1.0, 1.0, n**3))
    if complex_part:
        off = complex_part * np.ones(n**3 - 1)
        matrix = matrix.astype(complex) + 1j * sp.diags(off, 1) - 1j * sp.diags(off, -1)
    return matrix.tocsr()


class ArpackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = _laplacian_matrix(9)
        self.reference = np.linalg.eigvalsh(self.matrix.toarray())

    def test_lowest_eigenpairs_match_dense_diagonalization(self) -> None:
        result = run_arpack(self.matrix, 8, settings=ArpackSettings(tolerance=1e-12))

        np.testing.assert_allclose(result.eigenvalues, self.reference[:8], atol=1e-10)
        residual = self.matrix @ result.vectors - result.vectors * result.eigenvalues
        self.assertLess(np.max(np.linalg.norm(residual, axis=0)), 1e-8)
        self.assertEqual(result.converged_states, 8)
        self.assertEqual(len(result.calls), 1)
        self.assertGreater(result.matvecs, 2 * 8 + 5 - 1)

    def test_complex_hermitian_operator_goes_through_arpack(self) -> None:
        matrix = _laplacian_matrix(7, complex_part=0.3)
        reference = np.linalg.eigvalsh(matrix.toarray())
        result = run_arpack(matrix, 6, settings=ArpackSettings(tolerance=1e-12))

        np.testing.assert_allclose(result.eigenvalues, reference[:6], atol=1e-10)
        self.assertTrue(np.iscomplexobj(result.vectors))

    def test_start_vector_is_lapack_dlarnv_and_continues_between_solves(self) -> None:
        stream = LapackRandom(ARPACK_START_SEED)
        expected = 2.0 * stream.uniform_0_1(5) - 1.0
        np.testing.assert_array_equal(
            arpack_start_vector(LapackRandom(ARPACK_START_SEED), 5, np.float64),
            expected,
        )
        complex_start = arpack_start_vector(LapackRandom(ARPACK_START_SEED), 3, complex)
        pairs = 2.0 * LapackRandom(ARPACK_START_SEED).uniform_0_1(6) - 1.0
        np.testing.assert_array_equal(complex_start.real, pairs[0::2])
        np.testing.assert_array_equal(complex_start.imag, pairs[1::2])

        first = run_arpack(self.matrix, 4)
        again = run_arpack(self.matrix, 4)
        np.testing.assert_array_equal(first.eigenvalues, again.eigenvalues)
        self.assertNotEqual(first.start_seed, ARPACK_START_SEED)
        continued = run_arpack(self.matrix, 4, start_seed=first.start_seed)
        np.testing.assert_allclose(continued.eigenvalues, first.eigenvalues, atol=1e-6)

    def test_matvec_budget_failure_is_parsec_ierr_402(self) -> None:
        with self.assertRaises(IterativeDiagonalizationError) as caught:
            run_arpack(self.matrix, 8, settings=ArpackSettings(maximum_matvec=20))
        self.assertEqual(caught.exception.ierr, 402)
        self.assertIn("Maximum_Matvec", str(caught.exception))

    def test_unconverged_solve_is_retried_five_times_then_fails(self) -> None:
        settings = ArpackSettings(tolerance=1e-14, restarts_per_state=1)
        with self.assertRaises(IterativeDiagonalizationError) as caught:
            # One restart for twelve pairs cannot converge them.
            run_arpack(self.matrix, 12, settings=replace(settings, restarts_per_state=1))
        self.assertEqual(caught.exception.ierr, 402)
        self.assertEqual(caught.exception.calls, 5)
        self.assertLess(caught.exception.converged, 12)

    def test_budget_rule_of_parsec(self) -> None:
        self.assertEqual(matvec_budget(30000, 100, 8), 30000)
        self.assertEqual(matvec_budget(30000, 25768, 8), 25768 * 8)


class DiaglaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.size = 9
        self.matrix = _laplacian_matrix(self.size, seed=3)
        self.reference = np.linalg.eigvalsh(self.matrix.toarray())
        self.neighbors = nearest_neighbor_rows(_box_grid(self.size))

    def test_lowest_eigenpairs_match_dense_diagonalization(self) -> None:
        tolerance = 1e-6
        for neighbors in (None, self.neighbors):
            with self.subTest(preconditioned=neighbors is not None):
                result = run_diagla(
                    self.matrix,
                    10,
                    settings=DiaglaSettings(tolerance=tolerance, neighbors=neighbors),
                )
                np.testing.assert_allclose(
                    result.eigenvalues, self.reference[:10], atol=1e-9
                )
                # Locked in ascending order with ||r_i|| * ned < tol * i.
                self.assertTrue(np.all(np.diff(result.eigenvalues) >= 0))
                limits = tolerance * np.arange(1, 11) / 10
                residual = (
                    self.matrix @ result.vectors - result.vectors * result.eigenvalues
                )
                self.assertTrue(np.all(np.linalg.norm(residual, axis=0) < 1.01 * limits))
                self.assertEqual(result.all_vectors.shape, (self.size**3, 15))
                gram = result.vectors.T @ result.vectors
                np.testing.assert_allclose(gram, np.eye(10), atol=1e-10)

    def test_warm_start_from_converged_vectors_needs_one_window_per_state_block(self) -> None:
        settings = DiaglaSettings(tolerance=1e-6, neighbors=self.neighbors)
        cold = run_diagla(self.matrix, 10, settings=settings)
        warm = run_diagla(self.matrix, 10, settings=settings, initial_vectors=cold.all_vectors)
        self.assertLess(warm.matvecs, cold.matvecs / 5)
        np.testing.assert_allclose(warm.eigenvalues, self.reference[:10], atol=1e-9)

    def test_complex_hermitian_operator(self) -> None:
        matrix = _laplacian_matrix(7, complex_part=0.3)
        reference = np.linalg.eigvalsh(matrix.toarray())
        neighbors = nearest_neighbor_rows(_box_grid(7))
        result = run_diagla(
            matrix, 6, settings=DiaglaSettings(tolerance=1e-7, neighbors=neighbors)
        )
        np.testing.assert_allclose(result.eigenvalues, reference[:6], atol=1e-10)

    def test_budget_exhaustion_is_info_1_and_ierr_401(self) -> None:
        eval_, _, _, nec, mv, info = diagla_diag(self.matrix, 10, matvec_budget=30)
        self.assertEqual(info, 1)
        self.assertLess(nec, 10)
        self.assertGreaterEqual(mv, 30)
        with self.assertRaises(IterativeDiagonalizationError) as caught:
            run_diagla(self.matrix, 10, settings=DiaglaSettings(maximum_matvec=30))
        self.assertEqual(caught.exception.ierr, 401)
        self.assertEqual(caught.exception.info, 1)

    def test_zero_initial_vectors_become_unit_vectors_of_the_ird_sequence(self) -> None:
        # One Davidson step with a budget of five products: the start block is
        # lwin = 5 unit vectors placed by ird = mod(ird*137 + i*207 + 71, n) + 1.
        n = self.size**3
        _, evec, _, _, mv, info = diagla_diag(self.matrix, 3, matvec_budget=5)
        self.assertEqual(mv, 5)
        rows = []
        ird = 0
        for i in range(1, 6):
            ird = (ird * 137 + i * 207 + 71) % n + 1
            rows.append(ird - 1)
        # The Ritz vectors of the first step live in the span of those units.
        support = np.flatnonzero(np.any(np.abs(evec[:, :5]) > 0, axis=1))
        self.assertTrue(set(support.tolist()) <= set(rows))


class PreconditionerAndGridTests(unittest.TestCase):
    def test_fdo5scl_is_half_the_vector_plus_half_the_neighbour_mean(self) -> None:
        grid = build_cluster_grid(GridSettings(spacing=1.0, radius=2.2, expansion_order=2))
        neighbors = nearest_neighbor_rows(grid)
        self.assertEqual(neighbors.shape, (grid.size, 6))
        # A boundary point of the sphere has neighbours outside (-1).
        self.assertTrue(np.any(neighbors < 0))
        rng = np.random.default_rng(1)
        vector = rng.standard_normal(grid.size)
        expected = np.empty_like(vector)
        for row in range(grid.size):
            total = sum(vector[j] for j in neighbors[row] if j >= 0)
            expected[row] = 0.5 * (vector[row] + total / 6.0)
        apply = fdo5scl_preconditioner(neighbors)
        np.testing.assert_allclose(apply(vector), expected, rtol=0, atol=1e-14)
        # Spinors: each grid block on its own.
        spinor = np.concatenate([vector, 2.0 * vector])
        np.testing.assert_allclose(
            apply(spinor), np.concatenate([expected, 2.0 * expected]), atol=1e-14
        )

    def test_neighbour_order_is_setup_f90(self) -> None:
        grid = _box_grid(3)
        neighbors = nearest_neighbor_rows(grid)
        centre = 13  # (1, 1, 1) of a 3x3x3 grid in C order
        self.assertEqual(neighbors[centre].tolist(), [4, 22, 10, 16, 12, 14])


class EigvalDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = _laplacian_matrix(8, seed=5)
        self.shifted = (self.matrix + sp.diags(0.01 * np.ones(8**3))).tocsr()
        self.reference = np.linalg.eigvalsh(self.matrix.toarray())

    def _settings(self, method: str, filtering: bool | None) -> EigvalSettings:
        return EigvalSettings(
            safety_buffer=6,
            initial_method=method,
            arpack=ArpackSettings(tolerance=1e-10),
            diagla=DiaglaSettings(tolerance=1e-7),
            chebyshev_filtering=filtering,
        )

    def test_without_filtering_the_solver_runs_in_every_iteration(self) -> None:
        for method in ("arpack", "diagla"):
            with self.subTest(method=method):
                settings = self._settings(method, False)
                first = solve_eigval(self.matrix, 4, settings=settings)
                second = solve_eigval(self.shifted, 4, settings=settings, state=first.state)
                self.assertEqual(first.solver_path, method)
                self.assertEqual(second.solver_path, method)
                self.assertFalse(second.restarted)
                self.assertEqual(second.state.solves_completed, 2)
                self.assertEqual(first.state.working_states, 10)
                np.testing.assert_allclose(first.eigenvalues, self.reference[:4], atol=1e-8)
                np.testing.assert_allclose(
                    second.eigenvalues, self.reference[:4] + 0.01, atol=1e-8
                )
                self.assertIsNotNone(second.iterative_counts)

    def test_diagla_reuses_its_vectors_and_arpack_its_random_stream(self) -> None:
        diagla = self._settings("diagla", False)
        first = solve_eigval(self.matrix, 4, settings=diagla)
        self.assertEqual(first.state.window_vectors.shape, (8**3, 5))
        second = solve_eigval(self.shifted, 4, settings=diagla, state=first.state)
        self.assertLess(second.diagla_result.matvecs, first.diagla_result.matvecs)

        arpack = self._settings("arpack", False)
        first = solve_eigval(self.matrix, 4, settings=arpack)
        second = solve_eigval(self.matrix, 4, settings=arpack, state=first.state)
        self.assertNotEqual(first.state.start_seed, second.state.start_seed)

    def test_with_filtering_the_solver_runs_once_then_subspace(self) -> None:
        for method in ("arpack", "diagla"):
            with self.subTest(method=method):
                settings = self._settings(method, True)
                first = solve_eigval(self.matrix, 4, settings=settings)
                second = solve_eigval(self.shifted, 4, settings=settings, state=first.state)
                self.assertEqual(first.solver_path, method)
                self.assertEqual(second.solver_path, "subspace")
                self.assertTrue(first.state.subspace.first_filter)
                self.assertIsNone(second.iterative_counts)
                np.testing.assert_allclose(
                    second.eigenvalues, self.reference[:4] + 0.01, atol=1e-6
                )

    def test_chebff_and_chebdav_cannot_switch_filtering_off(self) -> None:
        with self.assertRaisesRegex(ValueError, "always follows"):
            EigvalSettings(initial_method="chebff", chebyshev_filtering=False)
        self.assertTrue(EigvalSettings(initial_method="chebdav").later_subspace_filter)
        self.assertFalse(EigvalSettings(initial_method="arpack").later_subspace_filter)


class ControlTests(unittest.TestCase):
    def test_dynamic_tolerance_rule(self) -> None:
        # min(0.1, 0.4/10) = 0.04; then the floor Diag_Tolerance.
        self.assertAlmostEqual(dynamic_diagonalization_tolerance(0.1, (0.4,), 1e-4), 0.04)
        self.assertAlmostEqual(dynamic_diagonalization_tolerance(0.04, (2.0,), 1e-4), 0.04)
        self.assertAlmostEqual(
            dynamic_diagonalization_tolerance(0.04, (0.3, 0.02), 1e-4), 0.002
        )
        self.assertEqual(dynamic_diagonalization_tolerance(0.002, (1e-7,), 1e-4), 1e-4)

    def test_new_settings_fields_are_hidden_from_repr_at_defaults(self) -> None:
        defaults = EigensolverSettings()
        self.assertNotIn("chebyshev_filtering", repr(defaults))
        self.assertNotIn("dynamic_tolerance", repr(defaults))
        self.assertNotIn("maximum_matvec", repr(defaults))
        self.assertTrue(repr(defaults).endswith("lanczos_steps=5, random_seed=7)"))
        shown = EigensolverSettings(
            method="arpack",
            chebyshev_filtering=True,
            dynamic_tolerance=True,
            maximum_matvec=100,
        )
        for text in ("chebyshev_filtering=True", "dynamic_tolerance=True", "maximum_matvec=100"):
            self.assertIn(text, repr(shown))
        scf = SCFSettings()
        self.assertNotIn("convergence_criterion_approach", repr(scf))
        self.assertEqual(scf.approach_criterion, 100.0 * scf.convergence_criterion)
        approach = SCFSettings(convergence_criterion_approach=0.05)
        self.assertIn("convergence_criterion_approach=0.05", repr(approach))
        self.assertEqual(approach.approach_criterion, 0.05)

    def test_filtering_rules_of_usrinputfile(self) -> None:
        self.assertTrue(EigensolverSettings(method="chebff").filters_later_iterations)
        self.assertFalse(EigensolverSettings(method="arpack").filters_later_iterations)
        self.assertTrue(
            EigensolverSettings(
                method="diagla", chebyshev_filtering=True
            ).filters_later_iterations
        )
        with self.assertRaisesRegex(ValueError, "cannot be False"):
            EigensolverSettings(method="chebdav", chebyshev_filtering=False)
        with self.assertRaisesRegex(ValueError, "subspace buffer"):
            EigensolverSettings(
                method="arpack", chebyshev_filtering=True, subspace_buffer=2
            )
        # Chebdav_Degree is not validated for ARPACK, DIAGLA or TEST mode.
        for method in ("arpack", "diagla", "none"):
            self.assertEqual(
                EigensolverSettings(method=method, first_filter_degree=5).method, method
            )


class InputTests(unittest.TestCase):
    def parse(self, replacements: dict[str, str]):
        text = H2_INPUT.read_text(encoding="utf-8")
        for old, new in replacements.items():
            self.assertIn(old, text)
            text = text.replace(old, new)
        with patch.object(Path, "read_text", return_value=text):
            return parse_parsec_input(H2_INPUT)

    def test_arpack_and_diagla_with_and_without_filtering(self) -> None:
        arpack = self.parse({"Eigensolver: chebff": "Eigensolver: arpack"})
        self.assertFalse(arpack.problem.eigensolver.filters_later_iterations)
        # Chebdav_Degree 10 is neither reset nor validated for ARPACK.
        self.assertEqual(arpack.problem.eigensolver.first_filter_degree, 20)
        self.assertTrue(any("Chebdav_Degree is not read" in w for w in arpack.warnings))
        filtered = self.parse(
            {
                "Eigensolver: chebff": "Eigensolver: diagla\nChebyshev_Filtering: .true.",
                "Chebdav_Degree: 10\n": "",
            }
        )
        self.assertTrue(filtered.problem.eigensolver.filters_later_iterations)
        self.assertTrue(any("refuses it" in w for w in filtered.warnings))
        none = self.parse({"Eigensolver: chebff": "Eigensolver: none"})
        self.assertEqual(none.problem.eigensolver.method, "none")
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "removed"):
            self.parse({"Eigensolver: chebff": "Eigensolver: trlanc"})

    def test_controls(self) -> None:
        translation = self.parse(
            {
                "Diag_Tolerance: 1.0e-4": (
                    "Diag_Tolerance: 1.0e-4\nDynamic_Diag_Tol: .true.\n"
                    "Convergence_Criterion_Approach: 0.5 eV\nMaximum_Matvec: 50000\n"
                    "Fix_Number_Eigenvalues: .false.\nExp_Comm: .true.\n"
                    "Max_Lap_Buffers: 4"
                )
            }
        )
        problem = translation.problem
        self.assertTrue(problem.eigensolver.dynamic_tolerance)
        self.assertEqual(problem.eigensolver.maximum_matvec, 50000)
        self.assertAlmostEqual(
            problem.scf.convergence_criterion_approach, 0.5 / 13.605693122994, places=6
        )
        self.assertTrue(any("Exp_Comm and Max_Lap_Buffers" in w for w in translation.warnings))
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Restart_Run"):
            self.parse(
                {"Diag_Tolerance: 1.0e-4": "Diag_Tolerance: 1.0e-4\nFix_Number_Eigenvalues: .true."}
            )
        # Chebyshev_Filtering false cannot switch the filter of CHEBFF off.
        chebff = self.parse(
            {"Eigensolver: chebff": "Eigensolver: chebff\nChebyshev_Filtering: .false."}
        )
        self.assertIsNone(chebff.problem.eigensolver.chebyshev_filtering)
        self.assertTrue(any("is ignored" in w for w in chebff.warnings))

    def test_accelerated_cli_routes_new_solvers_to_the_reference(self) -> None:
        from parsec_python.acceleration.cli import _reference_only_features

        # DIAGLA always, ARPACK where the CuPy eigensolvers are not selected
        # (the device ARPACK: acceleration/tests/test_geigen_parity.py).
        for replacement, phrase, device in (
            ("Eigensolver: arpack", "ARPACK", False),
            ("Eigensolver: diagla", "DIAGLA", False),
            ("Eigensolver: diagla", "DIAGLA", True),
        ):
            with self.subTest(replacement=replacement, device=device):
                translation = self.parse({"Eigensolver: chebff": replacement})
                features = _reference_only_features(
                    translation,
                    accelerated_backend_selected=False,
                    device_eigensolver_selected=device,
                )
                self.assertTrue(any(phrase in item for item in features))
        # The TEST mode and Dynamic_Diag_Tol run on the accelerated path,
        # ARPACK with the CuPy eigensolvers.
        for replacement, device in (
            ("Eigensolver: none", False),
            ("Eigensolver: chebff\nDynamic_Diag_Tol: .true.", False),
            ("Eigensolver: arpack", True),
        ):
            with self.subTest(replacement=replacement, device=device):
                translation = self.parse({"Eigensolver: chebff": replacement})
                self.assertEqual(
                    _reference_only_features(
                        translation,
                        accelerated_backend_selected=False,
                        device_eigensolver_selected=device,
                    ),
                    [],
                )
        plain = self.parse(
            {"Max_Iter: 50": "Max_Iter: 50\nConvergence_Criterion_Approach: 0.01"}
        )
        self.assertEqual(
            _reference_only_features(plain, accelerated_backend_selected=False), []
        )


class OutputTests(unittest.TestCase):
    def parse(self, replacement: str):
        text = H2_INPUT.read_text(encoding="utf-8").replace(
            "Eigensolver: chebff", replacement
        )
        with patch.object(Path, "read_text", return_value=text):
            return parse_parsec_input(H2_INPUT)

    def header(self, replacement: str) -> str:
        from parsec_python.Output import ParsecTextReporter

        written: list[str] = []
        ParsecTextReporter(written.append, self.parse(replacement)).header()
        return "\n".join(written)

    def test_solver_lines_of_usrinputfile(self) -> None:
        arpack = self.header("Eigensolver: arpack")
        self.assertIn(" Using ARPACK (implicitly restarted Arnoldi eigensolver)", arpack)
        self.assertIn(" No Chebyshev subspace filtering", arpack)
        self.assertNotIn("Lanczos/ARPACK", arpack)
        diagla = self.header("Eigensolver: diagla\nChebyshev_Filtering: .true.")
        self.assertIn(" Using DIAGLA (block preconditioned Lanczos eigen-problem solver)", diagla)
        self.assertIn(" Performing Chebyshev subspace filtering", diagla)
        self.assertIn(" Polynomial degree for Chebyshev filtering is     10", diagla)
        none = self.header("Eigensolver: none")
        self.assertIn("Preparing for Special TEST MODE", none)
        controls = self.header(
            "Eigensolver: chebff\nDynamic_Diag_Tol: .true.\n"
            "Convergence_Criterion_Approach: 0.01"
        )
        self.assertIn(" YOU ARE USING DYNAMIC TOLERANCE FOR CONVERGENCE", controls)
        self.assertIn(
            " Self-consistency convergence indicator is at    1.0000000000000000E-02  Ry",
            controls,
        )
        # Nothing of the kind for a calculation without the new keywords.
        plain = self.header("Eigensolver: chebff")
        for text in ("DYNAMIC", "indicator", "ARPACK", "DIAGLA"):
            self.assertNotIn(text, plain)

    def test_iteration_reports_matvecs_and_dynamic_tolerance(self) -> None:
        from parsec_python.Output import ParsecTextReporter
        from parsec_python.models import EnergyBreakdown, SCFIteration

        energy = EnergyBreakdown(
            eigenvalue=-1.0,
            hartree=0.5,
            integral_vxc_rho=-0.5,
            exchange_correlation=-0.4,
            electron_ion=-2.0,
            ion_ion=0.3,
            electronic=-2.0,
            total=-1.7,
        )
        written: list[str] = []
        reporter = ParsecTextReporter(written.append, self.parse("Eigensolver: arpack"))
        for iteration in (1, 2):
            reporter.iteration(
                SCFIteration(
                    iteration=iteration,
                    weighted_residual=0.02,
                    plain_residual=0.1,
                    eigen_residual_max=float("nan"),
                    hartree_residual=0.0,
                    energies=energy,
                    eigenvalues=(-0.5,),
                    occupations=(1.0,),
                    eigensolver_counts=((22, 6143),),
                    next_eigensolver_tolerance=0.002,
                )
            )
        text = "\n".join(written)
        self.assertIn(
            " Number of converged eigen-pairs per representation:          22", text
        )
        self.assertIn(" Number of matrix-vector multiplications:         6143", text)
        self.assertIn(" Cumulated number of matrix-vector mult. =        12286", text)
        self.assertIn("Dynamic tolerance is used in eigensolver\n\nNew tolerance:    0.00200", text)


class SCFTests(unittest.TestCase):
    """Hydrogen atom on a coarse grid: every solver gives the same SCF."""

    def _problem(self, eigensolver: EigensolverSettings, **scf) -> SinglePointInput:
        potential = DATA / "H_POTRE.DAT"
        return SinglePointInput(
            atoms=[Atom("H", [0.0, 0.0, 0.0])],
            pseudopotentials={"H": SpeciesPotential(potential, 0)},
            grid=GridSettings(spacing=0.8, radius=4.0, expansion_order=2),
            scf=SCFSettings(
                max_iterations=scf.pop("max_iterations", 30),
                number_of_states=2,
                convergence_criterion=1e-6,
                **scf,
            ),
            eigensolver=eigensolver,
        )

    def test_solvers_agree(self) -> None:
        energies = {}
        for name, eigensolver in (
            ("chebdav", EigensolverSettings(method="chebdav", tolerance=1e-8)),
            ("arpack", EigensolverSettings(method="arpack", tolerance=1e-10)),
            (
                "arpack_filter",
                EigensolverSettings(
                    method="arpack", tolerance=1e-10, chebyshev_filtering=True
                ),
            ),
            ("diagla", EigensolverSettings(method="diagla", tolerance=1e-8)),
        ):
            result = run_scf(prepare_single_point(self._problem(eigensolver)))
            self.assertTrue(result.converged, name)
            energies[name] = (result.energies.total, result.eigenvalues[0])
            counts = result.history[1].eigensolver_counts
            if name in ("arpack", "diagla"):
                self.assertEqual(counts[0][0], 2 + 6)
            else:
                self.assertEqual(counts, ())
        for name, (total, lowest) in energies.items():
            self.assertAlmostEqual(total, energies["chebdav"][0], places=6, msg=name)
            self.assertAlmostEqual(lowest, energies["chebdav"][1], places=5, msg=name)

    def test_dynamic_tolerance_is_recorded_and_floors_at_diag_tolerance(self) -> None:
        eigensolver = EigensolverSettings(
            method="arpack", tolerance=1e-6, dynamic_tolerance=True
        )
        result = run_scf(prepare_single_point(self._problem(eigensolver)))
        tolerances = [step.next_eigensolver_tolerance for step in result.history]
        expected = 0.1
        for step, tolerance in zip(result.history, tolerances):
            expected = max(min(expected, step.weighted_residual / 10), 1e-6)
            self.assertAlmostEqual(tolerance, expected)
        self.assertEqual(tolerances[-1], 1e-6)

    def test_approach_criterion_lowers_the_later_degree(self) -> None:
        degrees: list[int] = []

        def recording_solver(operator, states, *, settings, state):
            degrees.append(settings.subspace.polynomial_degree)
            return solve_eigval(operator, states, settings=settings, state=state)

        def run(approach):
            degrees.clear()
            system = prepare_single_point(
                self._problem(
                    EigensolverSettings(method="chebff"),
                    max_iterations=12,
                    convergence_criterion_approach=approach,
                )
            )
            result = run_scf(system, eigenproblem_solver=recording_solver)
            return list(degrees), result

        default_degrees, default_result = run(None)
        loose_degrees, loose_result = run(10.0)
        # A loose approach criterion lowers the degree from the seventh
        # iteration on (iter > 5 after the sixth).
        self.assertGreater(len(loose_degrees), 7)
        self.assertEqual(
            loose_degrees,
            [max(10, 15 - max(0, k - 6)) for k in range(1, len(loose_degrees) + 1)],
        )
        # The default 100 * vconv is the former hard-coded rule.
        weighted = [step.weighted_residual for step in default_result.history]
        degree = 15
        expected = []
        for iteration, residual in enumerate(weighted, start=1):
            expected.append(degree)
            if iteration > 5 and residual < 100 * 1e-6 and degree > 10:
                degree -= 1
        self.assertEqual(default_degrees, expected)


class KPointTests(unittest.TestCase):
    """Complex Bloch Hamiltonians (k != 0): ARPACK and DIAGLA as CHEBFF."""

    def test_kpoint_scf_agrees_between_solvers(self) -> None:
        from parsec_python.Grid import monkhorst_pack_grid
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.models import HartreeSettings, PeriodicCell, PeriodicGridSettings

        # 13 points per axis: an odd periodic grid (see the next test).
        box = 6.5
        cell = PeriodicCell(lattice_vectors=box * np.eye(3))
        energies = {}
        for name, eigensolver in (
            ("chebff", EigensolverSettings(method="chebff", tolerance=1e-6)),
            ("arpack", EigensolverSettings(method="arpack", tolerance=1e-10)),
            ("diagla", EigensolverSettings(method="diagla", tolerance=1e-7)),
        ):
            problem = SinglePointInput(
                atoms=[Atom("H", [box / 2, box / 2, box / 2])],
                pseudopotentials={"H": SpeciesPotential(DATA / "H_POTRE.DAT", 0)},
                grid=PeriodicGridSettings(spacing=0.5, expansion_order=6),
                periodic_cell=cell,
                scf=SCFSettings(
                    max_iterations=60, number_of_states=4, convergence_criterion=1e-5
                ),
                hartree=HartreeSettings(multipole_order=4),
                eigensolver=eigensolver,
            )
            system = prepare_periodic_single_point(problem)
            k_points, weights = monkhorst_pack_grid(cell, (2, 1, 1))
            self.assertGreater(np.max(np.abs(k_points)), 0.0)
            result = run_scf_kpoints(system, k_points, weights)
            self.assertTrue(result.converged, name)
            energies[name] = result.energies.total
        for name in ("arpack", "diagla"):
            self.assertAlmostEqual(energies[name], energies["chebff"], delta=1e-8, msg=name)

    def test_fdo5scl_annihilates_the_alternating_mode_of_an_even_periodic_grid(self) -> None:
        from parsec_python.Grid import build_periodic_grid
        from parsec_python.models import PeriodicCell, PeriodicGridSettings

        grid = build_periodic_grid(
            PeriodicCell(lattice_vectors=6.0 * np.eye(3)),
            PeriodicGridSettings(spacing=0.5, expansion_order=6),
        )
        alternating = (-1.0) ** grid.integer_coordinates.sum(axis=1)
        apply = fdo5scl_preconditioner(nearest_neighbor_rows(grid))
        self.assertEqual(np.max(np.abs(apply(alternating))), 0.0)


if __name__ == "__main__":
    unittest.main()
