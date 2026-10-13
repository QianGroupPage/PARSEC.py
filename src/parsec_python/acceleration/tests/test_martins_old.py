"""Old Martins pseudopotentials and ``occup.in`` on the accelerated paths.

Complete calculations of the accelerated driver are compared with the
``parsec.out`` of Fortran PARSEC for inputs with an old-format species:
SiNa4 of the Fortran example ``examples/tests/relaxation``, displaced
structures whose ``Core_Cutoff_Radius`` lies inside the Si projector (also
with occupations set by hand in ``occup.in``), and Ge2H2 with a core-corrected
old-format file.  The projector builder must end the projectors of such a
species at the radius of the input, and the reference cache key must tell
inputs apart that differ only in what they add to an old-format file.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from parsec_python.Input import parse_parsec_input
from parsec_python.V_ion import build_nonlocal_projectors
from parsec_python.acceleration import driver
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.tests.force_reference import assert_matches_fortran
from parsec_python.tests.martins_old_reference import (
    load_cases,
    support_sizes,
    write_case,
)


def _hybrid_available() -> bool:
    return cupy_available() and native_available()


@unittest.skipUnless(native_available(), "native extension is unavailable")
class NativeProjectorSupportTests(unittest.TestCase):
    def test_projectors_end_at_the_core_radius_of_the_input(self) -> None:
        from parsec_python.SCF import prepare_single_point
        from parsec_python.acceleration.V_ion import NativeIonicBuilders

        case = load_cases()["sina4_shifted_core_1.9"]
        with tempfile.TemporaryDirectory() as directory:
            translation = parse_parsec_input(write_case(case, Path(directory)))
            system = prepare_single_point(translation.problem)
        self.assertEqual(support_sizes(system), case["nonlocal_block_sizes"])
        arguments = (
            system.grid,
            system.atoms,
            system.pseudopotentials,
            translation.problem.pseudopotentials,
        )
        native = NativeIonicBuilders().build_nonlocal_projectors(*arguments)
        readable = build_nonlocal_projectors(*arguments)
        self.assertEqual(native.labels, readable.labels)
        np.testing.assert_array_equal(native.projectors.indptr, readable.projectors.indptr)
        np.testing.assert_array_equal(native.projectors.indices, readable.projectors.indices)
        np.testing.assert_allclose(
            native.projectors.data, readable.projectors.data, rtol=0, atol=1e-13
        )

    def test_cache_key_reads_the_input_of_an_old_format_species(self) -> None:
        cases = load_cases()

        def key(name: str, occupation_text: str | None = None) -> str:
            # One directory for every input, so that only its text differs.
            directory = Path(self._directory.name)
            path = write_case(cases[name], directory)
            if occupation_text is not None:
                (directory / "occup.in").write_text(occupation_text)
            return driver._reference_cache_key(
                parse_parsec_input(path).problem,
                SimpleNamespace(finite_difference_builder="native"),
                defer_native_laplacian=False,
                cache_directory=None,
            )

        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        wide = key("sina4_shifted")
        narrow = key("sina4_shifted_core_1.9")
        self.assertNotEqual(wide, narrow)
        self.assertEqual(narrow, key("sina4_shifted_core_1.9"))
        # Three full levels and two half-filled ones hold the same electrons.
        self.assertNotEqual(
            narrow, key("sina4_shifted_core_1.9", "title\n 3\n 2\n 4 0.5\n 5 0.5\n")
        )


@unittest.skipUnless(_hybrid_available(), "CUDA runtime or native extension is unavailable")
class FortranReferenceTests(unittest.TestCase):
    """Energy, levels, occupations and force tables of Fortran PARSEC."""

    def _run(self, name: str, *, operations: int, **options) -> None:
        case = load_cases()[name]
        with tempfile.TemporaryDirectory() as directory:
            translation = parse_parsec_input(write_case(case, Path(directory)))
            system = driver.prepare_single_point(translation.problem, **options)
        system.materialize_final_wavefunctions = True
        result = driver.run_scf(system)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.energies.total, case["energy"], delta=3.0e-8)
        for term, value in case["energies"].items():
            self.assertAlmostEqual(
                getattr(result.energies, term), value, delta=2.0e-7, msg=term
            )
        np.testing.assert_allclose(
            result.eigenvalues, case["eigenvalues"], rtol=0, atol=2.0e-7
        )
        np.testing.assert_allclose(
            result.occupations, case["occupations"], rtol=0, atol=5.01e-5
        )
        forces = driver.compute_forces(system, result)
        assert_matches_fortran(self, case, result.energies.total, forces)
        self.assertEqual(forces.symmetry_operations, operations)

    def test_relaxation_example_structure(self) -> None:
        self._run("sina4", operations=4, symmetry="off")

    def test_relaxation_example_structure_in_symmetry_sectors(self) -> None:
        self._run("sina4", operations=4)

    def test_displaced_structure(self) -> None:
        self._run("sina4_shifted", operations=1)

    def test_core_radius_inside_the_projector(self) -> None:
        self._run("sina4_shifted_core_1.9", operations=1)

    def test_cation_with_occupations_set_by_hand(self) -> None:
        self._run("sina4_cation_fractional", operations=1)

    def test_core_corrected_old_format_file(self) -> None:
        self._run("ge2h2_old_format", operations=1)


if __name__ == "__main__":
    unittest.main()
