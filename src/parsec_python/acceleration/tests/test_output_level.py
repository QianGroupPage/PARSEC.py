"""Backend verbosity must not suppress SCF reporting or fallback warnings."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from parsec_python.Input import parse_parsec_input
from parsec_python.acceleration.Output import AcceleratedTextReporter
from parsec_python.acceleration.models import BackendStatistics


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
                         ('symmetry_cache_directory', 'disabled'),
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
                     'symmetry_cache_directory = disabled',
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

    def test_setup_says_which_stages_ran_beside_the_preparation(self):
        note = 'are not part of the preparation wall time'
        for level in (1, 2):
            with self.subTest(level=level):
                for setup in ((), (('ionic_setup', 'inline'),)):
                    reporter, messages = self.reporter(level)
                    system = self.system()
                    system.backend_info.details += setup
                    reporter.setup(system)
                    self.assertNotIn(note, '\n'.join(messages))
                reporter, messages = self.reporter(level)
                system = self.system()
                system.backend_info.details += (
                    ('ionic_setup', 'overlapped with symmetry and orbital setup'),
                    ('ionic_setup_seconds', '4.905000'), ('ionic_setup_wait_seconds', '0.250000'))
                reporter.setup(system)
                report = '\n'.join(messages)
                self.assertIn(note, report)
                # Under the setup timings of the reference report, before the backend block.
                self.assertLess(report.index(note), report.index('Acceleration backend:'))
                self.assertIn('Thread [sec] = 4.905000, wait at its join [sec] = 0.250000.', report)

    def test_finish_says_which_filter_graphs_were_recorded_where_the_setup_named_others(self):
        key, line = 'orbital_sector_filter_graphs', 'Sector filter graphs as recorded = '
        shared = 'one per block width and degree'

        def report(level, at_setup, at_finish):
            reporter, messages = self.reporter(level)
            system = self.system()
            if at_setup is not None:
                system.backend_info.details += ((key, at_setup),)
            reporter.setup(system)
            details = () if at_finish is None else ((key, at_finish),)
            reporter.finish(SimpleNamespace(backend=SimpleNamespace(selected='cupy', details=details),
                                            backend_statistics=BackendStatistics()), 1.0)
            return '\n'.join(messages)

        # The detailed setup prints the switch; a filter that recorded no graph, or other ones, is said.
        for recorded in ('none recorded', 'one per block of a plan'):
            self.assertIn(f' {line}{recorded}', report(2, shared, recorded))
        self.assertNotIn(line, report(2, shared, shared))
        # Nothing where the setup said nothing of the graphs: the short report, and a run without sectors.
        self.assertNotIn(line, report(1, shared, 'none recorded'))
        self.assertNotIn(line, report(2, None, None))

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
