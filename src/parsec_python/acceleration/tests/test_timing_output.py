"""Integrated timing/reporting checks for the accelerated SciPy path."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from parsec_python import parse_parsec_input

from parsec_python.acceleration.Output import AcceleratedTextReporter
from parsec_python.acceleration.Output import accelerated_output
from parsec_python.acceleration.driver import (
    prepare_single_point,
    profile_hamiltonian_components,
    run_scf,
)


REFERENCE_DATA = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "data"
)
SMOKE_INPUT = REFERENCE_DATA / "H_cli_smoke.in"


class AcceleratedTimingOutputTests(unittest.TestCase):
    def test_capture_repetitions_are_reported_for_the_calculation_only(self) -> None:
        # The process has captured and repeated before this calculation begins.
        counts = {"captures": 7, "repetitions": 2}
        with patch.object(accelerated_output, "capture_statistics", lambda: dict(counts)):
            system, reporter, messages = self._prepared_reporter()
            result = run_scf(system, callback=reporter.iteration)
            reporter.finish(result, result.timings.total_seconds)
            self.assertNotIn("CUDA graph capture", "\n".join(messages))
            # A second calculation of the process, in which three graphs are
            # captured and one capture is recorded again.
            system, reporter, messages = self._prepared_reporter()
            result = run_scf(system, callback=reporter.iteration)
            counts.update(captures=10, repetitions=3)
            reporter.finish(result, result.timings.total_seconds)
        report = "\n".join(messages)
        self.assertIn(" CUDA graph capture repetitions (this process) =            1", report)
        self.assertIn(" CUDA graphs captured (this process) =            3", report)

    def _prepared_reporter(self):
        translation = parse_parsec_input(SMOKE_INPUT)
        messages: list[str] = []
        reporter = AcceleratedTextReporter(messages.append, translation)
        reporter.header()
        system = prepare_single_point(translation.problem, backend="scipy")
        reporter.setup(system)
        return system, reporter, messages

    def test_production_run_reports_source_and_backend_totals_without_profile(self) -> None:
        system, reporter, messages = self._prepared_reporter()

        # Normal eigensolver production must not enter the opt-in component
        # profiler or synchronize between each Hamiltonian term.
        with (
            patch.object(
                system.backend,
                "profile_components",
                side_effect=AssertionError("production entered component profile"),
            ) as profile,
            patch.object(
                system.backend,
                "synchronize",
                wraps=system.backend.synchronize,
            ) as synchronize,
        ):
            result = run_scf(system, callback=reporter.iteration)
        profile.assert_not_called()
        synchronize.assert_not_called()

        reporter.finish(result, result.timings.total_seconds)
        report = "\n".join(messages)

        for label in (
            "Finite-difference construction",
            "Local ionic potential setup",
            "Nonlocal ionic projector setup",
            "Initial Hamiltonian component timings",
            "Finite-difference (-nabla^2)",
            "V_ion diagonal/local",
            "V_ion nonlocal projectors",
            "Initial V_xc (CA-LDA)",
            "Initial exchange-correlation",
            "Diagonalization subtotal",
            "Hartree potential subtotal",
            "Exchange-correlation subtotal",
            "Accelerated Hamiltonian statistics",
            "H applications",
            "Orbital vectors applied",
            "Total H application time",
        ):
            with self.subTest(output_label=label):
                self.assertIn(label, report)

        self.assertNotIn("Profile finite_difference_apply", report)
        self.assertEqual(result.backend_statistics.component_profile_seconds, {})
        self.assertGreater(result.backend_statistics.applications, 0)
        self.assertGreater(result.backend_statistics.vectors_applied, 0)

        history = result.history
        self.assertAlmostEqual(
            result.timings.diagonalization_seconds,
            sum(item.diagonalization_seconds for item in history),
        )
        self.assertAlmostEqual(
            result.timings.hartree_seconds,
            sum(item.hartree_seconds for item in history),
        )
        self.assertAlmostEqual(
            result.timings.xc_seconds,
            sum(item.xc_seconds for item in history),
        )
        self.assertGreaterEqual(result.timings.initial_xc_seconds, 0.0)

    def test_component_breakdown_is_reported_only_after_explicit_profile(self) -> None:
        system, reporter, messages = self._prepared_reporter()
        timings = profile_hamiltonian_components(
            system,
            block_size=2,
            repeats=1,
        )
        self.assertEqual(
            set(timings),
            {
                "finite_difference_apply",
                "local_potential_apply",
                "nonlocal_potential_apply",
            },
        )
        self.assertTrue(all(value >= 0.0 for value in timings.values()))

        result = run_scf(system, callback=reporter.iteration)
        reporter.finish(result, result.timings.total_seconds)
        report = "\n".join(messages)
        for name in timings:
            self.assertIn(f"Profile {name} [sec]", report)


if __name__ == "__main__":
    unittest.main()
