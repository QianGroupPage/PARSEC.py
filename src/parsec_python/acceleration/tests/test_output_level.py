"""Backend verbosity must not suppress SCF reporting or fallback warnings."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from parsec_python.Input import parse_parsec_input
from parsec_python.acceleration.Output import AcceleratedTextReporter


DATA = Path(__file__).resolve().parents[2] / 'tests/data/H_cli_smoke.in'


class BackendOutputLevelTests(unittest.TestCase):
    def reporter(self, level):
        messages = []
        translation = replace(parse_parsec_input(DATA), output_level=level)
        reporter = AcceleratedTextReporter(messages.append, translation)
        reporter.reference = Mock()
        return reporter, messages

    def system(self):
        return SimpleNamespace(
            backend_info=SimpleNamespace(
                requested='auto', selected='cupy', dtype='float64', device='CUDA:0',
                implementation='Detailed kernel implementation',
                details=(('finite_difference_builder', 'native'),
                         ('hartree_backend', 'native'),
                         ('native_openmp_max_threads', '28'),
                         ('orbital_sector_later_filter_precision', 'float64'),
                         ('orbital_symmetry', 'CuPy real representations'),
                         ('gpu_kernel_configuration', 'verbose kernel settings'),
                         ('backend_resolution_seconds', '0.123')),
                fallback_reasons=('example fallback warning',)),
            backend=SimpleNamespace(statistics=SimpleNamespace(initialization_seconds=0.5)))

    def test_level_one_keeps_hybrid_summary_and_warnings(self):
        reporter, messages = self.reporter(1)
        system = self.system()
        reporter.setup(system)
        report = '\n'.join(messages)
        for text in ('Selected backend  = cupy', 'hartree_backend = native',
                     'native_openmp_max_threads = 28',
                     'orbital_sector_later_filter_precision = float64',
                     'example fallback warning', 'Output_Level: 2'):
            self.assertIn(text, report)
        self.assertNotIn('Implementation    =', report)
        self.assertNotIn('gpu_kernel_configuration', report)
        self.assertNotIn('backend_resolution_seconds', report)
        reporter.reference.setup.assert_called_once_with(system)

    def test_levels_two_and_higher_keep_all_details(self):
        for level in (2, 4, 6):
            with self.subTest(level=level):
                reporter, messages = self.reporter(level)
                system = self.system()
                reporter.setup(system)
                report = '\n'.join(messages)
                self.assertIn('Implementation    = Detailed kernel implementation', report)
                for key, value in system.backend_info.details:
                    self.assertIn(f'{key} = {value}', report)
                self.assertIn('example fallback warning', report)
                self.assertNotIn('set Output_Level', report)

    def test_scf_iterations_delegate_unchanged_at_both_levels(self):
        for level in (1, 2):
            reporter, _ = self.reporter(level)
            step = object()
            reporter.iteration(step)
            reporter.reference.iteration.assert_called_once_with(step)

    def test_missing_output_level_defaults_to_one(self):
        self.assertEqual(parse_parsec_input(DATA).output_level, 1)


if __name__ == '__main__':
    unittest.main()
