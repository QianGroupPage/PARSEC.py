"""The eigensolver, SCF-control and start-vector options of the PARSEC 1.5 round.

Labels of ``Input/solver_options.py``, the seed of the start vectors (``Vector_Rng_Seed``), PARSEC 1.5's
residual-driven CHEBFF, the reorthogonalization threshold, 1.5's Lanczos bounds and SUBSPACE lower
bound, the degree-decrease iteration, and that none of them changes anything at its default.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python.Eigensolvers.chebff import ChebFFResidualError, ChebFFSettings, run_chebff
from parsec_python.Eigensolvers.lapack_random import (
    PARSEC_RANDOM_ARRAY_SEED,
    LapackRandom,
    lapack_seed_from_integer,
)
from parsec_python.Eigensolvers.orthogonalize import orthonormalize
from parsec_python.Eigensolvers.spectral_bounds import (
    lanczos_bound_full_orthogonalization,
    lanczos_bound_parsec15,
    lanczos_upper_bound,
    spectral_bound,
)
from parsec_python.Eigensolvers.subspace import (
    SubspaceSettings,
    SubspaceState,
    parsec15_lower_bound,
    run_subspace_filter,
)
from parsec_python.Input import parse_parsec_input
from parsec_python.Input.parsec_input import ParsecInputError, UnsupportedParsecOptionError
from parsec_python.Input.routines import NEUTRAL_KEYS
from parsec_python.Input.solver_options import SOLVER_KEYS, read_solver_options
from parsec_python.models import EigensolverSettings, SCFSettings
from parsec_python.SCF.single_point import eigval_settings_for, solver_option_fields

DATA = Path(__file__).parents[1] / "data"
H2_INPUT = DATA / "H2_parsec.in"


def _test_matrix(size: int = 120, seed: int = 3) -> np.ndarray:
    """A symmetric matrix with a few low, separated levels and a dense upper spectrum."""

    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((size, size)))
    values = np.concatenate([[-2.0, -1.5, -1.2, -1.0], np.linspace(1.0, 40.0, size - 4)])
    return (q * values) @ q.T


class SeedTests(unittest.TestCase):
    def test_seed_map_gives_valid_distinct_lapack_seeds(self) -> None:
        seen = set()
        for value in (*range(0, 200), -1, -7, 2**40, 12345):
            seed = lapack_seed_from_integer(value)
            self.assertEqual(len(seed), 4)
            self.assertTrue(all(0 <= digit < 4096 for digit in seed))
            self.assertEqual(seed[-1] % 2, 1)
            self.assertEqual(LapackRandom(seed).seed, seed)
            seen.add(seed)
        self.assertEqual(len(seen), 204)
        self.assertEqual(lapack_seed_from_integer(5), lapack_seed_from_integer(5))

    def test_default_settings_keep_the_canonical_seed_and_text(self) -> None:
        settings = EigensolverSettings()
        self.assertEqual(settings.basis_seed, PARSEC_RANDOM_ARRAY_SEED)
        text = repr(settings)
        for name in (
            "vector_seed",
            "reorthogonalization_threshold",
            "first_filter_residual_test",
            "exact_state_count",
            "subspace_lower_bound",
            "scaled_subspace_filter",
            "lanczos_bounds",
            "lanczos_bound_steps",
        ):
            self.assertNotIn(name, text)
        # The default is 4 (PARSEC 1.5's iter > 4) since 2026-11-04; 5, the
        # earlier rule, is printed.
        self.assertNotIn("degree_decrease_after", repr(SCFSettings()))
        self.assertIn("degree_decrease_after=5", repr(SCFSettings(degree_decrease_after=5)))
        seeded = EigensolverSettings(vector_seed=9, random_seed=9)
        self.assertIn("vector_seed=9", repr(seeded))
        self.assertEqual(seeded.basis_seed, lapack_seed_from_integer(9))
        # The option fields of a default step are empty: the solver settings
        # are those of every earlier version.
        self.assertEqual(solver_option_fields(settings, 1.0e-4, 15), ({}, {}, {}))
        default = eigval_settings_for(settings, 15)
        self.assertEqual(default.chebff, ChebFFSettings(
            polynomial_degree=20, filter_cycles=3, lanczos_steps=10, block_size=6, random_seed=7
        ))
        self.assertEqual(default.subspace.lower_bound_rule, "monotone")

    def test_seed_changes_the_start_basis_of_chebff(self) -> None:
        matrix = _test_matrix()
        base = run_chebff(matrix, 8, settings=ChebFFSettings(filter_cycles=1))
        again = run_chebff(matrix, 8, settings=ChebFFSettings(filter_cycles=1))
        seeded = run_chebff(
            matrix,
            8,
            settings=ChebFFSettings(filter_cycles=1, basis_seed=lapack_seed_from_integer(4)),
        )
        np.testing.assert_array_equal(base.vectors, again.vectors)
        self.assertFalse(np.array_equal(base.vectors, seeded.vectors))


class ResidualDrivenChebffTests(unittest.TestCase):
    def test_early_exit_stop_and_continue(self) -> None:
        matrix = _test_matrix()
        loose = run_chebff(
            matrix, 8, settings=ChebFFSettings(filter_cycles=6, residual_tolerance=50.0), residual_states=4
        )
        self.assertEqual(len(loose.cycles), 1)
        self.assertLess(loose.cycles[0].maximum_residual, 50.0)
        with self.assertRaises(ChebFFResidualError) as caught:
            run_chebff(
                matrix, 8, settings=ChebFFSettings(filter_cycles=2, residual_tolerance=1.0e-14),
                residual_states=4,
            )
        self.assertIn("ierr 401", str(caught.exception))
        self.assertEqual(caught.exception.cycles, 2)
        kept = run_chebff(
            matrix,
            8,
            settings=ChebFFSettings(filter_cycles=2, residual_tolerance=1.0e-14, residual_failure="continue"),
            residual_states=4,
        )
        self.assertEqual(len(kept.cycles), 2)
        # Without the test the cycles carry no residual and match the plain run.
        plain = run_chebff(matrix, 8, settings=ChebFFSettings(filter_cycles=2))
        self.assertIsNone(plain.cycles[-1].maximum_residual)
        np.testing.assert_array_equal(plain.vectors, kept.vectors)

    def test_threshold_parameter(self) -> None:
        rng = np.random.default_rng(1)
        block = rng.standard_normal((200, 6))
        block[:, 5] = block[:, 4] + 1.0e-9 * rng.standard_normal(200)
        default = orthonormalize(block, rng=LapackRandom()).basis
        explicit = orthonormalize(block, rng=LapackRandom(), reorthogonalization_threshold=0.1).basis
        np.testing.assert_array_equal(default, explicit)
        strict = orthonormalize(block, rng=LapackRandom(), reorthogonalization_threshold=0.68).basis
        error = np.abs(strict.T @ strict - np.eye(6)).max()
        self.assertLess(error, 1.0e-13)
        with self.assertRaises(ValueError):
            orthonormalize(block, reorthogonalization_threshold=1.0)


class BoundTests(unittest.TestCase):
    def test_parsec15_bounds(self) -> None:
        matrix = _test_matrix()
        largest = np.linalg.eigvalsh(matrix)[-1]
        bound = lanczos_bound_parsec15(matrix, steps=12, rng=np.random.default_rng(2))
        self.assertEqual(bound.steps, 12)
        self.assertGreater(bound.upper_bound, largest)
        self.assertAlmostEqual(bound.middle, 0.5 * (bound.ritz_values[-1] + bound.ritz_values[0]))
        old = lanczos_upper_bound(matrix, steps=12, rng=np.random.default_rng(2))
        self.assertEqual(old.steps, 8)  # the earlier PARSEC's clamp
        full = lanczos_bound_full_orthogonalization(matrix, steps=20, rng=np.random.default_rng(2))
        spectrum = np.linalg.eigvalsh(matrix)
        self.assertGreater(full.upper_bound, spectrum[-1] - 1.0e-8)
        self.assertLess(full.lower_bound, spectrum[0] + 1.0e-8)
        self.assertIs(type(spectral_bound(matrix, "lanczos", 10, rng=np.random.default_rng(0))), type(bound))
        with self.assertRaises(ValueError):
            spectral_bound(matrix, "static")

    def test_parsec15_lower_bound_rules(self) -> None:
        values = np.array([-1.0, 0.5, 2.0])
        bound = lanczos_bound_parsec15(_test_matrix(), steps=10, rng=np.random.default_rng(0))
        first = SubspaceState(operator_dimension=3, working_states=3, eigenvalues=values,
                              vectors=np.eye(3), filter_lower_bound=None, first_filter=True)
        lower, _ = parsec15_lower_bound(first, "adaptive", bound)
        self.assertAlmostEqual(lower, 0.99 * 2.0)
        lower, _ = parsec15_lower_bound(first, "smart", bound)
        self.assertAlmostEqual(lower, max(min(0.99 * 2.0, bound.middle), 0.0))
        later = replace(first, first_filter=False, filter_lower_bound=1.0)
        self.assertAlmostEqual(parsec15_lower_bound(later, "smart", bound)[0], 2.0)
        # q = 2 > 1.3: the previous bound; q = 1.2: the mean; q < 1: r_max + 0.05.
        self.assertAlmostEqual(parsec15_lower_bound(later, "adaptive", bound)[0], 1.0)
        self.assertAlmostEqual(
            parsec15_lower_bound(replace(later, filter_lower_bound=2.0 / 1.2), "adaptive", bound)[0],
            0.5 * (2.0 / 1.2 + 2.0),
        )
        self.assertAlmostEqual(
            parsec15_lower_bound(replace(later, filter_lower_bound=4.0), "adaptive", bound)[0], 2.05
        )

    def test_subspace_with_parsec15_rules_runs(self) -> None:
        matrix = _test_matrix()
        first = run_chebff(matrix, 8, settings=ChebFFSettings(filter_cycles=2))
        state = SubspaceState(
            operator_dimension=matrix.shape[0], working_states=8,
            eigenvalues=first.eigenvalues, vectors=first.vectors,
        )
        for settings in (
            SubspaceSettings(lower_bound_rule="smart", bounds="lanczos", bound_steps=10),
            SubspaceSettings(lower_bound_rule="adaptive", scaled_filter=True),
            SubspaceSettings(reorthogonalization_threshold=0.68),
        ):
            result = run_subspace_filter(matrix, state, settings=settings)
            again = run_subspace_filter(matrix, result.state, settings=settings)
            np.testing.assert_allclose(again.eigenvalues[:4], np.linalg.eigvalsh(matrix)[:4], atol=1.0e-6)


class InputTests(unittest.TestCase):
    def parse(self, replacements: dict[str, str]):
        text = H2_INPUT.read_text(encoding="utf-8")
        for old, new in replacements.items():
            self.assertIn(old, text)
            text = text.replace(old, new)
        with patch.object(Path, "read_text", return_value=text):
            return parse_parsec_input(H2_INPUT)

    def add(self, lines: str):
        return self.parse({"Mixing_Method: Anderson": f"Mixing_Method: Anderson\n{lines}"})

    def test_labels_keep_the_default_routines(self) -> None:
        # Both sets of routines share the eigensolvers and SCF loops.
        self.assertTrue(SOLVER_KEYS <= NEUTRAL_KEYS)
        # Since 2026-11-04: the
        # controls of the earlier parity round and the MPI tuning labels act the same
        # under both sets, so writing them chooses no routines.
        from parsec_python.Input.routines import decide_routines

        for label in (
            "convergence_criterion_approach",
            "chebyshev_filtering",
            "maximum_matvec",
            "fix_number_eigenvalues",
            "mpi_groups",
            "exp_comm",
            "max_lap_buffers",
        ):
            with self.subTest(label=label):
                self.assertIn(label, NEUTRAL_KEYS)
                self.assertFalse(decide_routines({label: [object()]}, {}, requested=False).fortran)

    def test_seed_labels(self) -> None:
        problem = self.add("Vector_Rng_Seed: 12345").problem
        self.assertEqual(problem.eigensolver.vector_seed, 12345)
        self.assertEqual(problem.eigensolver.random_seed, 12345)
        clock = self.add("Vector_Rng_Seed: -1")
        self.assertGreater(clock.problem.eigensolver.vector_seed, 0)
        self.assertTrue(any("from the clock" in w for w in clock.warnings))
        inert = self.add("Vector_Rng_Type: 2\nOld_Vector_Random: .false.\nThread_Random_Init: .true.")
        self.assertIsNone(inert.problem.eigensolver.vector_seed)
        self.assertTrue(any("Vector_Rng_Type and Old_Vector_Random" in w for w in inert.warnings))
        # The default input is untouched.
        self.assertEqual(repr(self.parse({}).problem.eigensolver), repr(self.add("Rritz_Logic: none").problem.eigensolver))

    def test_subspace_rules_and_bounds(self) -> None:
        problem = self.add("Smart_Subspace: .true.\nChebyshev_Scaled: .true.\nBounds_Iter: 12").problem
        self.assertEqual(problem.eigensolver.subspace_lower_bound, "smart")
        self.assertTrue(problem.eigensolver.scaled_subspace_filter)
        self.assertEqual(problem.eigensolver.lanczos_bounds, "lanczos")
        self.assertEqual(problem.eigensolver.lanczos_bound_steps, 12)
        self.assertEqual(self.add("Smart_Subspace: .false.").problem.eigensolver.subspace_lower_bound, "adaptive")
        self.assertEqual(self.add("Bounds_Method: lanczos_full").problem.eigensolver.lanczos_bounds, "lanczos_full")
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "uninitialized"):
            self.add("Bounds_Method: static")
        with self.assertRaisesRegex(ParsecInputError, "ierr 129"):
            self.add("Bounds_Method: chebyshev")
        options = eigval_settings_for(problem.eigensolver, 10)
        # max(degree/2, Bounds_Iter): 12 steps; 1.5's full Lanczos serves CHEBFF only.
        self.assertEqual((options.chebff.bounds, options.chebff.bound_steps), ("lanczos", 12))
        self.assertEqual((options.subspace.bounds, options.subspace.lower_bound_rule), ("lanczos", "smart"))
        self.assertTrue(options.subspace.scaled_filter)

    def test_filter_restart_labels(self) -> None:
        # Spectrum slicing, Rritz_Logic, Test_Logic and Thread_Random_Init are read
        # by the parser's own block of 1.5 labels (tests/unit/test_parsec15_input.py).
        restart = self.add("Filter_Restart_After: 4")
        self.assertTrue(any("relaxation and molecular" in w for w in restart.warnings))
        moving = self.add(
            "Filter_Restart_After: 4\nRestart_Cheb_Remained_Times: 2\nMinimization: bfgs\nMovement_Num: 3"
        )
        self.assertEqual(moving.problem.eigensolver.filter_restart, (4, 2, 3))

    def test_first_filter_rules_of_parsec15(self) -> None:
        self.assertEqual(self.parse({"Eigensolver: chebff": "Eigensolver: chebff\nFF_MaxIter: 12"})
                         .problem.eigensolver.first_filter_cycles, 12)
        # PARSEC 1.5 accepts FF_MaxIter 0 (its restart_run_just_check_force tests keep the
        # wavefunctions of restart.h5); without them here, the first solve runs 3 cycles.
        zero = self.parse({"Eigensolver: chebff": "Eigensolver: chebff\nFF_MaxIter: 0"})
        self.assertEqual(zero.problem.eigensolver.first_filter_cycles, 3)
        self.assertTrue(any("FF_MaxIter=0" in w and "restart.h5" in w for w in zero.warnings))
        reset = self.parse({"Eigensolver: chebff": "Eigensolver: chebff\nFF_MaxIter: 60"})
        # FF_MaxIter of CHEBFF defaults to 3 (PARSEC 1.5) since 2026-11-04 (2 before).
        self.assertEqual(reset.problem.eigensolver.first_filter_cycles, 3)
        low = self.parse({"Chebdav_Degree: 10": "Chebdav_Degree: 8"})
        self.assertEqual(low.problem.eigensolver.first_filter_degree, 12)
        negative = self.parse({"Diag_Tolerance: 1.0e-4": "Diag_Tolerance: -1"})
        self.assertEqual(negative.problem.eigensolver.tolerance, 1.0e-4)
        # PARSEC 1.5's working space of exactly States_Num vectors is the opt-in
        # Exact_State_Count; Subspace_Buffer_Size,
        # which 1.5 does not read, keeps the earlier meaning: 0..5 reset to 6.
        exact = self.add("Exact_State_Count: .true.").problem.eigensolver
        self.assertTrue(exact.exact_state_count)
        self.assertEqual((exact.subspace_buffer, exact.working_buffer), (0, 0))
        both = self.add("Exact_State_Count: .true.\nSubspace_Buffer_Size: 9")
        self.assertEqual(both.problem.eigensolver.working_buffer, 0)
        self.assertTrue(any("Subspace_Buffer_Size=9 is not used" in w for w in both.warnings))
        for buffer in (0, 3):
            with self.subTest(buffer=buffer):
                old = self.add(f"Subspace_Buffer_Size: {buffer}")
                self.assertFalse(old.problem.eigensolver.exact_state_count)
                self.assertEqual(old.problem.eigensolver.subspace_buffer, 6)
                self.assertIn(
                    f"Subspace_Buffer_Size={buffer} is below 6 for chebff filtering; PARSEC resets it to 6.",
                    old.warnings,
                )
        self.assertFalse(self.add("Exact_State_Count: .false.").problem.eigensolver.exact_state_count)

    def test_extensions(self) -> None:
        problem = self.add(
            "FF_Residual_Test: stop\nChebyshev_Degree_Decrease_After: 4\nReorthogonalization_Threshold: 0.68"
        ).problem
        self.assertEqual(problem.eigensolver.first_filter_residual_test, "stop")
        self.assertEqual(problem.scf.degree_decrease_after, 4)
        self.assertEqual(problem.eigensolver.reorthogonalization_threshold, 0.68)
        options = eigval_settings_for(problem.eigensolver, 10, tolerance=1.0e-3)
        self.assertEqual(options.chebff.residual_tolerance, 1.0e-3)
        self.assertEqual(options.chebff.residual_failure, "stop")
        self.assertEqual(options.subspace.reorthogonalization_threshold, 0.68)
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "no first filter"):
            self.parse({"Eigensolver: chebff": "Eigensolver: arpack\nFF_Residual_Test: stop"})
        with self.assertRaisesRegex(ParsecInputError, "between 0 and 1"):
            self.add("Reorthogonalization_Threshold: 1.5")
        # Diag_Tolerance <= 0 tests no residual (1.5: tol > 0 only).
        off = self.parse({"Diag_Tolerance: 1.0e-4": "Diag_Tolerance: 0\nFF_Residual_Test: continue"})
        self.assertEqual(off.problem.eigensolver.first_filter_residual_test, "off")

    def test_accelerated_cli_routes_the_readable_options(self) -> None:
        from parsec_python.acceleration.cli import _reference_only_solver_options

        self.assertEqual(_reference_only_solver_options(self.parse({}).problem), [])
        self.assertEqual(_reference_only_solver_options(self.add("Vector_Rng_Seed: 3").problem), [])
        self.assertEqual(
            _reference_only_solver_options(self.add("Chebyshev_Degree_Decrease_After: 4").problem), []
        )
        for line in ("FF_Residual_Test: stop", "Reorthogonalization_Threshold: 0.68", "Smart_Subspace: .true."):
            with self.subTest(line=line):
                self.assertEqual(len(_reference_only_solver_options(self.add(line).problem)), 1)


class CompleteInputTests(unittest.TestCase):
    """bak_complete_parsec.in (Input/complete_input.py) gives the same problem with the eigensolver and SCF labels."""

    def test_round_trip(self) -> None:
        import shutil
        import tempfile
        import warnings

        from parsec_python.Input.complete_input import COMPLETE_INPUT_NAME
        from parsec_python.Output.complete_input import write_complete_input

        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, True)
        shutil.copy2(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
        for extra in (
            "Vector_Rng_Seed: -1\nSmart_Subspace: .true.\nBounds_Iter: 12\nFF_Residual_Test: stop\n"
            "Chebyshev_Degree_Decrease_After: 4\nReorthogonalization_Threshold: 0.68\nExact_State_Count: .true.",
            "Diag_Tolerance: -1\nFF_Residual_Test: stop\nVector_Rng_Type: 1\nChebyshev_Scaled: .true.",
            "Filter_Restart_After: 2\nMinimization: bfgs\nMovement_Num: 2\nBounds_Method: lanczos_full",
        ):
            with self.subTest(extra=extra):
                text = H2_INPUT.read_text(encoding="utf-8")
                if extra.startswith("Diag_Tolerance"):
                    text = text.replace("Diag_Tolerance: 1.0e-4", "")
                text += "\nGenerate_Complete_Parsec_In: .true.\n" + extra + "\n"
                (directory / "parsec.in").write_text(text, encoding="utf-8")
                translation = parse_parsec_input(directory / "parsec.in")
                write_complete_input(translation)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    again = parse_parsec_input(directory / COMPLETE_INPUT_NAME)
                self.assertEqual(repr(again.problem), repr(translation.problem))


class FilterRestartTests(unittest.TestCase):
    def test_counters_of_parsec15(self) -> None:
        from parsec_python.SCF import SCFWarmStart
        from parsec_python.SCF.filter_restart import FilterRestart

        restart = FilterRestart.from_eigensolver(EigensolverSettings(filter_restart=(2, 3, 1)))
        carried = SCFWarmStart(
            input_potential=np.zeros(3), hartree_potential=np.zeros(3), eigensolver_state="S"
        )
        self.assertEqual(restart.warm_start_after_move(carried, 1), (carried, None))
        # Movement 2: the first restart, with the atomic density (a cold start).
        self.assertEqual(restart.warm_start_after_move(carried, 2), (None, "density"))
        kept, reset = restart.warm_start_after_move(carried, 4)
        self.assertEqual(reset, "eigensolver")
        self.assertIsNone(kept.eigensolver_state)
        self.assertIs(kept.input_potential, carried.input_potential)
        spin = replace(carried, eigensolver_state=("U", "D"))
        self.assertEqual(restart.warm_start_after_move(spin, 6)[0].eigensolver_state, (None, None))
        # Three restarts used: none at movement 8.
        self.assertEqual(restart.warm_start_after_move(carried, 8), (carried, None))
        self.assertIsNone(FilterRestart.from_eigensolver(EigensolverSettings()))


class SCFLoopTests(unittest.TestCase):
    """The options in the readable SCF loop: a hydrogen atom on a coarse grid."""

    def _run(self, eigensolver: EigensolverSettings, **scf):
        from parsec_python.models import Atom, GridSettings, SinglePointInput, SpeciesPotential
        from parsec_python.SCF import prepare_single_point, run_scf
        import parsec_python.SCF.single_point as single_point

        problem = SinglePointInput(
            atoms=[Atom("H", [0.0, 0.0, 0.0])],
            pseudopotentials={"H": SpeciesPotential(DATA / "H_POTRE.DAT", 0)},
            grid=GridSettings(spacing=0.8, radius=4.0, expansion_order=2),
            scf=SCFSettings(max_iterations=12, number_of_states=2, convergence_criterion=1e-7, **scf),
            eigensolver=eigensolver,
        )
        degrees = []
        original = single_point.eigval_settings_for

        def recording(settings, filter_degree, **keywords):
            degrees.append(filter_degree)
            return original(settings, filter_degree, **keywords)

        with patch.object(single_point, "eigval_settings_for", recording):
            result = run_scf(prepare_single_point(problem))
        return result, degrees

    def test_degree_decrease_seed_and_exact_count(self) -> None:
        chebff = EigensolverSettings(method="chebff", first_filter_degree=10, filter_degree=15)
        # An approach criterion that always holds: the degree falls from the
        # iteration after degree_decrease_after on.
        # The default is 4 (PARSEC 1.5) since 2026-11-04; 5 is the earlier rule.
        _, five = self._run(chebff, convergence_criterion_approach=1.0e3, degree_decrease_after=5)
        _, four = self._run(chebff, convergence_criterion_approach=1.0e3)
        self.assertEqual(five[:7], [15, 15, 15, 15, 15, 15, 14])
        self.assertEqual(four[:6], [15, 15, 15, 15, 15, 14])
        seeded, _ = self._run(replace(chebff, vector_seed=11, random_seed=11))
        plain, _ = self._run(chebff)
        self.assertAlmostEqual(seeded.energies.total, plain.energies.total, delta=1.0e-6)
        exact, _ = self._run(replace(chebff, subspace_buffer=0, exact_state_count=True))
        self.assertAlmostEqual(exact.energies.total, plain.energies.total, delta=1.0e-6)

    def test_residual_stop_in_the_loop(self) -> None:
        stop = EigensolverSettings(
            method="chebff", first_filter_degree=10, tolerance=1.0e-12, first_filter_residual_test="stop"
        )
        with self.assertRaises(ChebFFResidualError):
            self._run(stop)
        loose, _ = self._run(replace(stop, tolerance=10.0))
        self.assertTrue(np.isfinite(loose.energies.total))


class ReaderTests(unittest.TestCase):
    def test_clock_seed_is_the_clock(self) -> None:
        warnings: list[str] = []
        scalar = {"vector_rng_seed": [object()]}
        options = read_solver_options(
            scalar,
            lambda key, default=None: "-1",
            lambda key, default=False: default,
            integer=lambda value, label: int(value),
            real=lambda value, label: float(value),
            eigensolver_method="chebff",
            filtering=True,
            ionic_motion=False,
            warnings=warnings,
            clock=lambda: 1234.9,
        )
        self.assertEqual(options.eigensolver["vector_seed"], 1234)
        self.assertIn("seed 1234", warnings[0])


if __name__ == "__main__":
    unittest.main()
