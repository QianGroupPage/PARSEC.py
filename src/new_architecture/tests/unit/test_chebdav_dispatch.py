from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from new_architecture.Eigensolvers import eigval as policy
from new_architecture.Eigensolvers import ChebDavSettings, run_chebdav
from new_architecture.Eigensolvers.subspace import SubspaceState
from new_architecture.models import (
    Atom,
    EigensolverSettings,
    GridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from new_architecture.SCF import prepare_single_point, run_scf


def _fake_initial_solver(operator, wanted_states, *, settings):
    """Return the common state shape accepted by eigval's first-solve path."""

    dimension = operator.shape[0]
    values = np.arange(wanted_states, dtype=float)
    vectors = np.eye(dimension, wanted_states)
    state = SimpleNamespace(
        operator_dimension=dimension,
        wanted_states=wanted_states,
        eigenvalues=values,
        vectors=vectors,
    )
    return SimpleNamespace(
        eigenvalues=values,
        vectors=vectors,
        residual_norms=np.zeros(wanted_states),
        state=state,
    )


def _fake_subspace(operator, state, *, settings):
    values = np.asarray(state.eigenvalues) + 0.25
    vectors = np.asarray(state.vectors)
    next_state = SubspaceState(
        operator_dimension=state.operator_dimension,
        working_states=state.working_states,
        eigenvalues=values,
        vectors=vectors,
        filter_lower_bound=float(values[-1]),
        first_filter=False,
        filters_completed=state.filters_completed + 1,
    )
    return SimpleNamespace(
        eigenvalues=values,
        vectors=vectors,
        residual_norms=np.full(state.working_states, 0.125),
        state=next_state,
    )


class ChebDavDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = policy.EigvalSettings(
            safety_buffer=2,
            initial_method="chebdav",
        )

    def test_explicit_chebdav_dispatch_does_not_call_chebff(self) -> None:
        operator = np.eye(12)
        with (
            patch.object(
                policy,
                "run_chebdav",
                side_effect=_fake_initial_solver,
            ) as chebdav,
            patch.object(policy, "run_chebff") as chebff,
            patch.object(policy, "run_subspace_filter") as later,
        ):
            result = policy.solve_eigval(
                operator,
                3,
                settings=self.settings,
            )

        chebdav.assert_called_once()
        chebff.assert_not_called()
        later.assert_not_called()
        self.assertEqual(chebdav.call_args.args[1], 5)
        self.assertEqual(result.solver_path, "chebdav")
        self.assertEqual(result.eigenvalues.shape, (3,))
        self.assertEqual(result.state.subspace.eigenvalues.shape, (5,))

    def test_chebdav_state_is_reused_by_exactly_one_subspace_filter(self) -> None:
        operator = np.eye(12)
        with (
            patch.object(
                policy,
                "run_chebdav",
                side_effect=_fake_initial_solver,
            ) as chebdav,
            patch.object(policy, "run_chebff") as chebff,
            patch.object(
                policy,
                "run_subspace_filter",
                side_effect=_fake_subspace,
            ) as later,
        ):
            initial = policy.solve_eigval(
                operator,
                3,
                settings=self.settings,
            )
            updated = policy.solve_eigval(
                2.0 * operator,
                3,
                settings=self.settings,
                state=initial.state,
            )

        chebdav.assert_called_once()
        chebff.assert_not_called()
        later.assert_called_once()
        self.assertEqual(updated.solver_path, "subspace")
        self.assertEqual(updated.state.solves_completed, 2)
        self.assertEqual(updated.state.subspace.filters_completed, 1)

    def test_incompatible_state_restarts_the_selected_chebdav_path(self) -> None:
        operator = np.eye(12)
        with (
            patch.object(
                policy,
                "run_chebdav",
                side_effect=_fake_initial_solver,
            ) as chebdav,
            patch.object(policy, "run_chebff") as chebff,
            patch.object(policy, "run_subspace_filter") as later,
        ):
            initial = policy.solve_eigval(
                operator,
                2,
                settings=self.settings,
            )
            restarted = policy.solve_eigval(
                operator,
                3,
                settings=self.settings,
                state=initial.state,
            )

        self.assertEqual(chebdav.call_count, 2)
        chebff.assert_not_called()
        later.assert_not_called()
        self.assertEqual(restarted.solver_path, "chebdav")
        self.assertTrue(restarted.restarted)
        self.assertEqual(
            restarted.restart_reason,
            "requested_state_count_changed",
        )

    def test_changing_initial_method_invalidates_the_saved_subspace(self) -> None:
        operator = np.eye(12)
        chebff_settings = policy.EigvalSettings(
            safety_buffer=2,
            initial_method="chebff",
        )
        with (
            patch.object(
                policy,
                "run_chebff",
                side_effect=_fake_initial_solver,
            ) as chebff,
            patch.object(
                policy,
                "run_chebdav",
                side_effect=_fake_initial_solver,
            ) as chebdav,
            patch.object(policy, "run_subspace_filter") as later,
        ):
            initial = policy.solve_eigval(
                operator,
                3,
                settings=chebff_settings,
            )
            restarted = policy.solve_eigval(
                operator,
                3,
                settings=self.settings,
                state=initial.state,
            )

        chebff.assert_called_once()
        chebdav.assert_called_once()
        later.assert_not_called()
        self.assertEqual(initial.state.initial_method, "chebff")
        self.assertEqual(restarted.state.initial_method, "chebdav")
        self.assertEqual(restarted.solver_path, "chebdav")
        self.assertTrue(restarted.restarted)
        self.assertEqual(restarted.restart_reason, "initial_solver_changed")


class ChebDavExecutionTests(unittest.TestCase):
    def test_small_diagonal_problem_converges_natively(self) -> None:
        diagonal = np.array([-4.0, -1.0, 0.5, 2.0, 4.0, 7.0, 10.0, 15.0])
        operator = np.diag(diagonal)

        result = run_chebdav(operator, 2)

        self.assertEqual(result.eigenvalues.shape, (2,))
        self.assertEqual(result.vectors.shape, (operator.shape[0], 2))
        np.testing.assert_allclose(result.eigenvalues, diagonal[:2], atol=1.0e-5)
        np.testing.assert_allclose(
            result.vectors.T @ result.vectors,
            np.eye(2),
            atol=2.0e-11,
        )
        residuals = operator @ result.vectors - (
            result.vectors * result.eigenvalues[None, :]
        )
        self.assertLess(float(np.max(np.linalg.norm(residuals, axis=0))), 2.0e-3)

    def test_outer_restart_uses_parsec_pre_deflation_dimensions(self) -> None:
        generator = np.random.default_rng(0)
        operator = generator.normal(size=(30, 30))
        operator = 0.5 * (operator + operator.T)

        result = run_chebdav(
            operator,
            8,
            settings=ChebDavSettings(
                block_size=3,
                workspace_window=6,
                convergence_tolerance=1.0e-8,
                random_seed=100,
            ),
        )

        # On iteration three, three pairs lock.  PARSEC tests nconv+hsizerst
        # before subtracting that locked prefix, so 3+9 > (14-3) triggers an
        # outer restart and retains 8-3=5 active vectors.
        third = result.iterations[2]
        self.assertEqual((third.locked_before, third.locked_after), (0, 3))
        self.assertTrue(third.outer_restart)
        self.assertEqual(third.active_dimension_retained, 5)
        self.assertEqual(result.state.outer_restarts, 2)

    def test_iteration_limit_cleanup_excludes_just_appended_block(self) -> None:
        generator = np.random.default_rng(3)
        operator = generator.normal(size=(30, 30))
        operator = 0.5 * (operator + operator.T)

        result = run_chebdav(
            operator,
            6,
            settings=ChebDavSettings(
                block_size=3,
                workspace_window=6,
                convergence_tolerance=1.0e-16,
                max_iterations=1,
                random_seed=100,
            ),
        )

        self.assertTrue(result.state.approximate_cleanup_used)
        self.assertEqual(result.state.truly_converged_states, 0)
        self.assertEqual(result.state.iterations_completed, 2)
        # Source num_mv: Lanczos 5 + initial filter 60 + two*(3+60)
        # + final six-vector filter 120.  The appended third block and the
        # cleanup H application are deliberately absent from this counter.
        self.assertEqual(result.state.matrix_vector_products, 311)

    def test_one_iteration_scf_accepts_chebdav(self) -> None:
        potential = Path(__file__).parents[1] / "data" / "H_POTRE.DAT"
        problem = SinglePointInput(
            atoms=[Atom("H", [0.0, 0.0, 0.0])],
            pseudopotentials={"H": SpeciesPotential(potential, 0)},
            grid=GridSettings(
                spacing=0.8,
                radius=3.0,
                expansion_order=2,
            ),
            scf=SCFSettings(
                max_iterations=1,
                number_of_states=2,
            ),
            eigensolver=EigensolverSettings(method="chebdav"),
        )

        result = run_scf(prepare_single_point(problem))

        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.eigenvalues.shape, (2,))
        self.assertTrue(np.all(np.isfinite(result.eigenvalues)))
        self.assertTrue(np.all(np.isfinite(result.density)))
        self.assertTrue(np.isfinite(result.energies.total))


if __name__ == "__main__":
    unittest.main()
