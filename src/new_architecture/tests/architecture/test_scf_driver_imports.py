"""Architecture checks for the readable SCF module boundary."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from new_architecture import driver
from new_architecture import SCF
from new_architecture.SCF import single_point as scf_implementation


class SCFDriverImportTests(unittest.TestCase):
    def test_scf_package_exports_the_numerical_implementation(self) -> None:
        self.assertIs(
            SCF.PreparedSinglePointSystem,
            scf_implementation.PreparedSinglePointSystem,
        )
        self.assertIs(
            SCF.prepare_single_point,
            scf_implementation.prepare_single_point,
        )
        self.assertIs(SCF.run_scf, scf_implementation.run_scf)

    def test_complete_driver_composes_prepare_then_scf(self) -> None:
        problem = object()
        prepared = object()
        result = object()
        callback = object()

        with (
            patch.object(
                driver,
                "prepare_single_point",
                return_value=prepared,
            ) as prepare,
            patch.object(driver, "run_scf", return_value=result) as solve,
        ):
            self.assertIs(
                driver.run_single_point(problem, callback=callback),
                result,
            )

        prepare.assert_called_once_with(problem)
        solve.assert_called_once_with(prepared, callback=callback)


if __name__ == "__main__":
    unittest.main()
