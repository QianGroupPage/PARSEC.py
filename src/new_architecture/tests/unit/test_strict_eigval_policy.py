from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from new_architecture.Eigensolvers import eigval as policy
from new_architecture.Eigensolvers.subspace import SubspaceState


def _fake_chebff(operator, wanted_states, *, settings):
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


class StrictEigvalPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = policy.EigvalSettings(safety_buffer=2)

    def test_first_call_uses_chebff_and_saves_all_working_states(self) -> None:
        operator = np.eye(12)
        with (
            patch.object(policy, "run_chebff", side_effect=_fake_chebff) as first,
            patch.object(policy, "run_subspace_filter") as later,
        ):
            result = policy.solve_eigval(
                operator,
                3,
                settings=self.settings,
            )

        first.assert_called_once()
        later.assert_not_called()
        self.assertEqual(first.call_args.args[1], 5)
        self.assertEqual(result.solver_path, "chebff")
        self.assertFalse(result.restarted)
        self.assertIsNone(result.restart_reason)
        self.assertEqual(result.eigenvalues.shape, (3,))
        self.assertEqual(result.vectors.shape, (12, 3))
        self.assertEqual(result.state.subspace.eigenvalues.shape, (5,))
        self.assertTrue(result.state.subspace.first_filter)

    def test_second_compatible_call_uses_exactly_one_subspace_filter(self) -> None:
        operator = np.eye(12)
        with (
            patch.object(policy, "run_chebff", side_effect=_fake_chebff) as first,
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

        first.assert_called_once()
        later.assert_called_once()
        self.assertEqual(updated.solver_path, "subspace")
        self.assertFalse(updated.restarted)
        self.assertEqual(updated.state.solves_completed, 2)
        self.assertEqual(updated.state.subspace.filters_completed, 1)
        np.testing.assert_allclose(updated.residual_norms, 0.125)

    def test_changed_state_count_restarts_chebff_instead_of_reusing_subspace(self) -> None:
        operator = np.eye(12)
        with (
            patch.object(policy, "run_chebff", side_effect=_fake_chebff) as first,
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

        self.assertEqual(first.call_count, 2)
        later.assert_not_called()
        self.assertEqual(restarted.solver_path, "chebff")
        self.assertTrue(restarted.restarted)
        self.assertEqual(
            restarted.restart_reason,
            "requested_state_count_changed",
        )

    def test_subspace_failure_is_propagated_without_solver_fallback(self) -> None:
        operator = np.eye(12)
        with patch.object(policy, "run_chebff", side_effect=_fake_chebff):
            initial = policy.solve_eigval(
                operator,
                3,
                settings=self.settings,
            )

        with (
            patch.object(policy, "run_chebff") as fallback,
            patch.object(
                policy,
                "run_subspace_filter",
                side_effect=RuntimeError("filter failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "filter failed"):
                policy.solve_eigval(
                    operator,
                    3,
                    settings=self.settings,
                    state=initial.state,
                )

        fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
