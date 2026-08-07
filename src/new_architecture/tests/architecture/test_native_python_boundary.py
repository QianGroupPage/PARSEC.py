from __future__ import annotations

import ast
from pathlib import Path
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PACKAGE_DIRECTORIES = (
    "Input",
    "Grid",
    "Laplacian",
    "Pseudopotential",
    "V_ion",
    "Hartree",
    "V_xc",
    "Hamiltonian",
    "Eigensolvers",
    "Occupations",
    "Mixer",
    "Energy",
    "SCF",
    "Output",
    "provenance",
)
FORBIDDEN_RUNTIME_IMPORTS = {
    "cffi",
    "ctypes",
    "f2py",
    "numpy.f2py",
    "subprocess",
}


class NativePythonBoundaryTests(unittest.TestCase):
    @staticmethod
    def _runtime_python_files() -> tuple[Path, ...]:
        files = list(PACKAGE_ROOT.glob("*.py"))
        for directory_name in RUNTIME_PACKAGE_DIRECTORIES:
            files.extend((PACKAGE_ROOT / directory_name).rglob("*.py"))
        return tuple(sorted(files))

    def test_runtime_never_imports_a_fortran_bridge_or_process_runner(self) -> None:
        violations: list[str] = []
        for path in self._runtime_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                imported: list[str] = []
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported = [node.module]
                for name in imported:
                    if name in FORBIDDEN_RUNTIME_IMPORTS:
                        violations.append(f"{path.relative_to(PACKAGE_ROOT)}: {name}")
        self.assertEqual(
            violations,
            [],
            "the calculation must remain a native Python implementation",
        )

    def test_runtime_contains_no_wsl_or_parsec_executable_command(self) -> None:
        violations: list[str] = []
        command_markers = ("wsl.exe", "wsl bash", "parsec.exe", "mpirun parsec")
        for path in self._runtime_python_files():
            text = path.read_text(encoding="utf-8").lower()
            for marker in command_markers:
                if marker in text:
                    violations.append(
                        f"{path.relative_to(PACKAGE_ROOT)}: {marker}"
                    )
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
