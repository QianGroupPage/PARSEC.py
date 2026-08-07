from __future__ import annotations

import math
from pathlib import Path
import unittest

from new_architecture.models import (
    Atom,
    EigensolverSettings,
    GridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from new_architecture.SCF import prepare_single_point, run_scf


class StrictSCFEigensolverTests(unittest.TestCase):
    def _problem(
        self,
        *,
        method: str = "chebff",
        iterations: int = 2,
        first_filter_degree: int = 20,
    ) -> SinglePointInput:
        potential = Path(__file__).parents[1] / "data" / "H_POTRE.DAT"
        eigensolver = (
            EigensolverSettings(
                method="chebff",
                first_filter_degree=first_filter_degree,
            )
            if method == "chebff"
            else EigensolverSettings(
                method=method,
                subspace_buffer=0,
                matvec_block_size=1,
            )
        )
        return SinglePointInput(
            atoms=[Atom("H", [0.0, 0.0, 0.0])],
            pseudopotentials={"H": SpeciesPotential(potential, 0)},
            grid=GridSettings(
                spacing=0.8,
                radius=3.0,
                expansion_order=2,
            ),
            scf=SCFSettings(
                max_iterations=iterations,
                number_of_states=2,
            ),
            eigensolver=eigensolver,
        )

    def test_chebff_state_is_reused_by_later_subspace_filter(self) -> None:
        result = run_scf(prepare_single_point(self._problem()))

        self.assertEqual(result.iterations, 2)
        self.assertTrue(math.isnan(result.history[0].eigen_residual_max))
        self.assertTrue(math.isfinite(result.history[1].eigen_residual_max))

    def test_chebff_degree_ten_does_not_enter_chebdav_validation(self) -> None:
        result = run_scf(
            prepare_single_point(
                self._problem(iterations=1, first_filter_degree=10)
            )
        )

        self.assertEqual(result.iterations, 1)
        self.assertTrue(math.isnan(result.history[0].eigen_residual_max))

    def test_unported_solver_is_not_silently_substituted(self) -> None:
        system = prepare_single_point(
            self._problem(method="arpack", iterations=1)
        )

        with self.assertRaisesRegex(
            NotImplementedError,
            "not yet ported.*no alternate solver",
        ):
            run_scf(system)


if __name__ == "__main__":
    unittest.main()
