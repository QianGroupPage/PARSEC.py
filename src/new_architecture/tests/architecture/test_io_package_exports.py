"""Input and output package-export checks."""

from __future__ import annotations

import importlib
import unittest


class InputOutputPackageExportTests(unittest.TestCase):
    def test_input_package_exports_its_implementation_objects(self) -> None:
        implementation = importlib.import_module(
            "new_architecture.Input.parsec_input"
        )
        package = importlib.import_module("new_architecture.Input")

        for name in implementation.__all__:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(package, name),
                    getattr(implementation, name),
                )

    def test_output_package_exports_its_implementation_objects(self) -> None:
        implementation = importlib.import_module(
            "new_architecture.Output.parsec_output"
        )
        package = importlib.import_module("new_architecture.Output")

        for name in implementation.__all__:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(package, name),
                    getattr(implementation, name),
                )


if __name__ == "__main__":
    unittest.main()
