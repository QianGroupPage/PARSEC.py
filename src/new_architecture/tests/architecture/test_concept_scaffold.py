from __future__ import annotations

import json
from pathlib import Path
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONCEPT_PACKAGES = (
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
REMOVED_FLAT_MODULES = (
    "energy.py",
    "finite_difference.py",
    "grid.py",
    "hamiltonian.py",
    "hartree.py",
    "ionic.py",
    "mixing.py",
    "occupations.py",
    "parsec_input.py",
    "parsec_output.py",
    "pseudopotential.py",
    "scf.py",
    "state.py",
    "xc.py",
)
ALLOWED_PACKAGE_STATUSES = {
    "scaffold_only",
    "in_progress",
    "implemented_unverified",
    "verified_component",
    "verified_integration",
    "out_of_scope",
}
ALLOWED_LITERAL_STATUSES = {
    "not_verified",
    "partially_verified",
    "verified",
    "out_of_scope",
}


class ConceptScaffoldTests(unittest.TestCase):
    def test_each_concept_is_an_explicit_python_package(self) -> None:
        for name in CONCEPT_PACKAGES:
            with self.subTest(package=name):
                package_dir = PACKAGE_ROOT / name
                self.assertTrue(package_dir.is_dir(), package_dir)
                self.assertTrue((package_dir / "__init__.py").is_file(), package_dir)

    def test_redundant_flat_compatibility_modules_are_absent(self) -> None:
        existing = [
            name for name in REMOVED_FLAT_MODULES if (PACKAGE_ROOT / name).exists()
        ]
        self.assertEqual(existing, [])

    def test_source_map_is_complete_and_uses_declared_statuses(self) -> None:
        manifest_path = PACKAGE_ROOT / "provenance" / "source_map.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema_version"], 2)
        self.assertFalse(manifest["calls_parsec_fortran"])

        entries = manifest["components"]
        mapped_packages = [entry["python_package"] for entry in entries]
        self.assertEqual(len(mapped_packages), len(set(mapped_packages)))
        self.assertEqual(
            set(mapped_packages),
            set(CONCEPT_PACKAGES) - {"provenance"},
        )

        for entry in entries:
            with self.subTest(component=entry["component"]):
                self.assertIn(
                    entry["concept_package_status"],
                    ALLOWED_PACKAGE_STATUSES,
                )
                self.assertIn(
                    entry["literal_parsec_status"],
                    ALLOWED_LITERAL_STATUSES,
                )
                self.assertIsInstance(entry["parsec_sources"], list)
                self.assertTrue(entry["parsec_sources"])
                self.assertIsInstance(entry["python_modules"], list)
                self.assertTrue(entry["python_modules"])
                for module_path in entry["python_modules"]:
                    self.assertTrue(
                        (PACKAGE_ROOT / module_path).is_file(),
                        module_path,
                    )
                self.assertIsInstance(entry["verification_evidence"], list)

    def test_unverified_claims_do_not_carry_verification_evidence(self) -> None:
        manifest_path = PACKAGE_ROOT / "provenance" / "source_map.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        for entry in manifest["components"]:
            if entry["literal_parsec_status"] == "not_verified":
                with self.subTest(component=entry["component"]):
                    self.assertEqual(entry["verification_evidence"], [])

    def test_architecture_document_states_native_port_boundary(self) -> None:
        architecture = (PACKAGE_ROOT / "ARCHITECTURE.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("does not invoke or bind to the PARSEC Fortran", architecture)
        self.assertIn("There is no silent ARPACK", architecture)


if __name__ == "__main__":
    unittest.main()
