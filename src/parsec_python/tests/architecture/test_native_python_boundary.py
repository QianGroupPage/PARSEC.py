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
    "Forces",
    "Relaxation",
    "MolecularDynamics",
    "Embedding",
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
# The one allowed exception, by module and import: the binding of libxc, the optional third-party
# library of exchange-correlation functionals the code connects on request (``LibXC_Functional``;
# MPL-2.0, loaded only when such a functional is asked for).  ctypes there calls a C library of
# functionals in this process; it is no Fortran bridge and no process runner, and the code's own
# functionals stay native Python.  Nothing else in that module (no subprocess, cffi, f2py) and no
# other module may use these imports.
ALLOWED_RUNTIME_IMPORTS = {
    ("V_xc/libxc_library.py", "ctypes"),
}


class NativePythonBoundaryTests(unittest.TestCase):
    @staticmethod
    def _runtime_python_files() -> tuple[Path, ...]:
        # ``main.py`` may re-execute the same native-Python solver in the
        # checkout's validated virtual environment.  It is process/bootstrap
        # code, not part of the numerical runtime audited by this test.
        files = [
            path for path in PACKAGE_ROOT.glob("*.py")
            if path.name != "main.py"
        ]
        for directory_name in RUNTIME_PACKAGE_DIRECTORIES:
            files.extend((PACKAGE_ROOT / directory_name).rglob("*.py"))
        return tuple(sorted(files))

    def test_runtime_never_imports_a_fortran_bridge_or_process_runner(self) -> None:
        violations: list[str] = []
        allowed_seen: set[tuple[str, str]] = set()
        for path in self._runtime_python_files():
            relative = path.relative_to(PACKAGE_ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                imported: list[str] = []
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported = [node.module]
                for name in imported:
                    if name in FORBIDDEN_RUNTIME_IMPORTS:
                        if (relative, name) in ALLOWED_RUNTIME_IMPORTS:
                            allowed_seen.add((relative, name))
                        else:
                            violations.append(f"{relative}: {name}")
        self.assertEqual(
            violations,
            [],
            "the calculation must remain a native Python implementation",
        )
        # The allowance names a module that exists and still needs it (a stale entry is removed).
        self.assertEqual(allowed_seen, ALLOWED_RUNTIME_IMPORTS)

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
