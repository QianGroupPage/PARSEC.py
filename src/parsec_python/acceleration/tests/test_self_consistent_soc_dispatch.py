"""CLI dispatch tests for the accelerated self-consistent SOC path.

These run on this (GPU-less) machine too: the dispatch logic itself
(``_accelerated_self_consistent_soc_eligible``, ``_reference_only_features``,
and ``main``'s delegate-vs-accelerate branch) is exercised end-to-end via the
"no CuPy available" path, which is exactly what ``--backend auto``/``scipy``
is supposed to do -- cleanly delegate to the reference CPU solver rather
than silently ignoring SOC. The actual CuPy spinor Hamiltonian/eigensolver
code path itself is only exercised on real hardware (see
``test_cupy_spinor.py``/``test_chebff_spinor.py`` for what *is* validated
here without a GPU).
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python.Input import parse_parsec_input

from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.cli import (
    _accelerated_self_consistent_soc_eligible,
    _reference_only_features,
    _resolve_symmetry_mode,
    main as accelerated_main,
)

GPU_AVAILABLE = cupy_available()

_AU_POTRE = (
    Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
)

_BASE_INPUT = """\
Boundary_Sphere_Radius: 6.0
Grid_Spacing: 0.6
Expansion_Order: 6

States_Num: 20
Net_Charges: 0
Fermi_Temp: 500.0
Max_Iter: 40
Convergence_Criterion: 5.0e-3

Mixing_Method: Anderson
Mixing_Param: 0.25
Eigensolver: chebff

SO_from_scratch: .true.

Atom_Types_Num: 1
Total_Atom_Num: 1

Atom_Type: Au
SO_PSP: .true.
Local_Component: s
Begin atom_coord
   0 0 0
End atom_coord

Correlation_Type: CA
"""


class EligibilityPredicateTests(unittest.TestCase):
    """Pure-logic checks, no CLI/grid/SCF involved."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.directory = Path(self._tmpdir.name)
        shutil.copy(_AU_POTRE, self.directory / "Au_POTRE.DAT")

    def _translation(self, extra: str = ""):
        path = self.directory / "parsec.in"
        path.write_text(_BASE_INPUT + extra, encoding="utf-8")
        return parse_parsec_input(path)

    def test_gamma_only_self_consistent_soc_is_eligible(self) -> None:
        translation = self._translation()
        self.assertTrue(_accelerated_self_consistent_soc_eligible(translation))

    def test_combined_with_spin_polarization_is_not_eligible(self) -> None:
        translation = self._translation(extra="Spin_Polarization: .true.\n")
        self.assertFalse(_accelerated_self_consistent_soc_eligible(translation))

    def test_periodic_is_not_eligible(self) -> None:
        translation = self._translation(
            extra=(
                "Periodic_System: .true.\n"
                "Boundary_Conditions: bulk\n"
                "begin Cell_Shape\n8.0 8.0 8.0\nend Cell_Shape\n"
            )
        )
        self.assertFalse(_accelerated_self_consistent_soc_eligible(translation))

    def test_plain_scalar_input_is_not_eligible(self) -> None:
        text = _BASE_INPUT.replace("SO_from_scratch: .true.\n", "").replace(
            "SO_PSP: .true.\n", ""
        )
        path = self.directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        translation = parse_parsec_input(path)
        self.assertFalse(_accelerated_self_consistent_soc_eligible(translation))

    def test_reference_only_features_empty_when_eligible_and_cupy_selected(self) -> None:
        translation = self._translation()
        self.assertEqual(
            _reference_only_features(translation, accelerated_backend_selected=True),
            [],
        )

    def test_reference_only_features_flags_when_cupy_not_selected(self) -> None:
        translation = self._translation()
        features = _reference_only_features(
            translation, accelerated_backend_selected=False
        )
        self.assertIn("self-consistent spin-orbit coupling (SO_from_scratch)", features)

    def test_reference_only_features_flags_spin_polarized_combo_regardless(self) -> None:
        translation = self._translation(extra="Spin_Polarization: .true.\n")
        features = _reference_only_features(
            translation, accelerated_backend_selected=True
        )
        self.assertEqual(len(features), 1)
        self.assertIn("spin polarization", features[0])

    def test_reference_only_features_flags_periodic_combo_once(self) -> None:
        translation = self._translation(
            extra=(
                "Periodic_System: .true.\n"
                "Boundary_Conditions: bulk\n"
                "begin Cell_Shape\n8.0 8.0 8.0\nend Cell_Shape\n"
                "Kpoint_Method: mp\n"
                "begin Monkhorst_Pack_Grid\n2 1 1\nend Monkhorst_Pack_Grid\n"
            )
        )
        features = _reference_only_features(
            translation, accelerated_backend_selected=True
        )
        # One periodic-SOC message, not a second redundant k-point message.
        self.assertEqual(len(features), 1)
        self.assertIn("periodic self-consistent spin-orbit coupling", features[0])


class SymmetryModeResolutionTests(unittest.TestCase):
    """Found via a real GPU run of examples/0d_AuH/reduced_self_consistent_soc:
    AuH has a nontrivial reflection symmetry that --symmetry auto happily
    detects and exploits for the scalar path, but self-consistent SOC's
    CuPySpinorHamiltonian/run_chebff_spinor only accept plain complex128
    ndarrays, not the compact SymmetryScalarField representation symmetry
    reduction produces -- crashing with
    "TypeError: unsupported operand type(s) for +: 'float' and
    'SymmetryScalarField'" at the very first SCF step. These tests cover the
    fix (force symmetry off for accelerated self-consistent SOC) without
    needing a GPU, since the resolution logic itself is pure."""

    def test_non_soc_requests_are_unaffected(self) -> None:
        for requested in (None, "auto", "on", "off"):
            with self.subTest(requested=requested):
                resolved = _resolve_symmetry_mode(
                    requested, False, accelerated_self_consistent_soc=False
                )
                expected = requested if requested is not None else "auto"
                self.assertEqual(resolved, expected)

    def test_soc_forces_off_when_unset_or_auto(self) -> None:
        for requested in (None, "auto"):
            with self.subTest(requested=requested):
                resolved = _resolve_symmetry_mode(
                    requested, False, accelerated_self_consistent_soc=True
                )
                self.assertEqual(resolved, "off")

    def test_soc_forces_off_when_explicitly_off(self) -> None:
        resolved = _resolve_symmetry_mode(
            "off", False, accelerated_self_consistent_soc=True
        )
        self.assertEqual(resolved, "off")

    def test_soc_rejects_explicit_on(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_symmetry_mode(
                "on", False, accelerated_self_consistent_soc=True
            )

    def test_ignore_symmetry_input_flag_does_not_bypass_the_soc_rejection(self) -> None:
        # Ignore_Symmetry=true only changes the *unset* default; an explicit
        # --symmetry on must still be rejected for SOC regardless of it.
        with self.assertRaises(ValueError):
            _resolve_symmetry_mode(
                "on", True, accelerated_self_consistent_soc=True
            )


class AcceleratedCliDelegationTests(unittest.TestCase):
    """End-to-end: on this GPU-less machine, --backend auto/scipy must still
    converge to the same result as the reference solver (via delegation),
    and --backend cupy must cleanly refuse rather than crash or silently
    run a scalar SCF."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.directory = Path(self._tmpdir.name)
        shutil.copy(_AU_POTRE, self.directory / "Au_POTRE.DAT")
        self.input_path = self.directory / "parsec.in"
        self.input_path.write_text(_BASE_INPUT, encoding="utf-8")

    @unittest.skipIf(GPU_AVAILABLE, "this check targets the no-GPU fallback")
    def test_backend_auto_delegates_and_converges(self) -> None:
        exit_code = accelerated_main([str(self.input_path), "--quiet"])
        self.assertEqual(exit_code, 0)
        archive = np.load(self.directory / "parsec_python_results.npz")
        self.assertTrue(bool(archive["converged"]))

    @unittest.skipIf(GPU_AVAILABLE, "this check targets the no-GPU fallback")
    def test_backend_cupy_explicit_is_refused_cleanly(self) -> None:
        exit_code = accelerated_main(
            [str(self.input_path), "--backend", "cupy", "--dry-run"]
        )
        self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
