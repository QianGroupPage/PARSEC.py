"""The default sphere of the input in the accelerated drivers.

Every MPI rank runs the rule on its own and the runner compares what they
resolved before any of them prepares; the record of the domain reaches the
details of the result and the timing file; and a calculation whose radius
the rule chose runs through the accelerated command line.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from parsec_python.Hartree import domain, valence_point_charges
from parsec_python.Input import parse_parsec_input
from parsec_python.V_ion import load_pseudopotentials
from parsec_python.acceleration import cli as cli_module
from parsec_python.acceleration.Output import AcceleratedTextReporter, accelerated_output
from parsec_python.acceleration.Output.accelerated_output import (
    RadiusEstimatesThread,
    domain_details,
    domain_report_thread_requested,
)
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.benchmarks import mpi_full_scf
from parsec_python.acceleration.models import BackendInfo, BackendStatistics
from parsec_python.models import Atom, GridSettings, HartreeSettings, SpeciesPotential
from parsec_python.tests import parsed_inputs
from parsec_python.tests.parsed_inputs import BASE_PSEUDOPOTENTIALS, NANODIAMOND


DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
# What the launch environment of a cluster names for its GPU runs.  A smoke
# run of this module names the SciPy backend and must not inherit them.
LAUNCH_BACKENDS = (
    "PARSEC_HARTREE_LINEAR_BACKEND",
    "PARSEC_HARTREE_BOUNDARY_BACKEND",
    "PARSEC_IONIC_BACKEND",
    "PARSEC_CUPY_RESIDENT_HARTREE",
)


def nanodiamond(radius: bool, folder: Path):
    text = NANODIAMOND.read_text(encoding="utf-8").replace("\r\n", "\n")
    path = folder / "parsec.in"
    path.write_bytes(
        (text if radius else parsed_inputs.without_radius(text)).encode("utf-8")
    )
    return parse_parsec_input(path, pseudopotential_directory=BASE_PSEUDOPOTENTIALS)


class RankAgreementTests(unittest.TestCase):
    def test_what_a_rank_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            chosen = nanodiamond(False, Path(folder))
            given = nanodiamond(True, Path(folder))
        self.assertEqual(
            mpi_full_scf._parsed_domain(chosen),
            (
                chosen.problem.grid.radius.hex(),
                (1.0e-3).hex(),
                ("16.8 ang", "1.0e-03 Ry"),
            ),
        )
        self.assertEqual(
            mpi_full_scf._parsed_domain(given),
            (given.problem.grid.radius.hex(), (1.0e-3).hex(), None),
        )

    def test_ranks_that_differ_stop_the_run(self) -> None:
        radius, tolerance = (31.747411546609168).hex(), (1.0e-3).hex()
        same = (radius, tolerance, ("16.8 ang", "1.0e-03 Ry"))
        mpi_full_scf._require_same_domain([same])
        mpi_full_scf._require_same_domain([same, same, same, same])
        for other in (
            ((31.93638423438698).hex(), tolerance, ("16.9 ang", "1.0e-03 Ry")),
            (radius, (9.4e-4).hex(), ("16.8 ang", "9.4e-04 Ry")),
            (radius, tolerance, None),
            # One unit in the last place of the radius.
            (float(np.nextafter(31.747411546609168, 40.0)).hex(), tolerance, same[2]),
        ):
            with self.assertRaisesRegex(
                ValueError,
                r"^the sphere radius or the Hartree boundary tolerance differs "
                r"across ranks: rank 0 has .* rank 2 has ",
            ):
                mpi_full_scf._require_same_domain([same, same, other, same])

    def test_runner_compares_the_ranks_before_it_prepares(self) -> None:
        from parsec_python.acceleration import driver

        asked = []

        class Comm:
            rank, size = 0, 2

            @staticmethod
            def allgather(value):
                asked.append(type(value).__name__)
                if isinstance(value, str):
                    return [value, value + "-other-node"]
                if isinstance(value, dict):
                    return [value, value]
                # The other rank resolved the next lattice value.
                return [value, (value[0], value[1], ("16.9 ang", "1.0e-03 Ry"))]

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder).resolve()
            text = NANODIAMOND.read_text(encoding="utf-8").replace("\r\n", "\n")
            (folder / "parsec.in").write_bytes(
                parsed_inputs.without_radius(text).encode("utf-8")
            )
            arguments = mpi_full_scf._build_parser().parse_args(
                [
                    "--input",
                    str(folder / "parsec.in"),
                    "--output-dir",
                    str(folder / "out"),
                    "--pp-dir",
                    str(BASE_PSEUDOPOTENTIALS),
                    "--devices",
                    "0",
                    "--quiet",
                ]
            )
            prepare = Mock()
            with (
                patch.object(driver, "prepare_single_point", prepare),
                patch.object(mpi_full_scf, "_source_provenance", lambda: None),
                self.assertRaisesRegex(ValueError, "differs across ranks"),
            ):
                mpi_full_scf.execute(arguments, None, Comm(), None, 0.0)
            prepare.assert_not_called()
            # Host names, then input digests, then the domain: nothing after.
            self.assertEqual(asked, ["str", "dict", "tuple"])
            self.assertFalse((folder / "out").exists())

    def test_timing_record_of_an_input_with_a_radius(self) -> None:
        # The serial control of the runner tests, whose preparation, SCF and
        # reporter are stand-ins, on the smoke input: no second of the rule,
        # and the boundary record what a comparison of two runs reads.
        from parsec_python.acceleration.tests import test_mpi_scf

        harness = test_mpi_scf.FullSCFRunnerContextTests(
            "test_switch_is_on_unless_named_off"
        )
        harness.setUp()
        self.addCleanup(harness.doCleanups)
        plan = SimpleNamespace(
            order=16, minimum_order=9, tolerance=1.0e-3, atomic_tail=True
        )
        with tempfile.TemporaryDirectory() as folder:
            record, _log, _cache = harness.run_serial_control(
                Path(folder).resolve() / "out", plan=plan
            )
        self.assertEqual(record["per_rank"][0]["domain_rule_seconds"], 0.0)
        self.assertEqual(
            sorted(record["result"]["hartree_boundary"]),
            [
                "atomic_tail",
                "atomic_tail_values",
                "kernel",
                "multipole_order",
                "solver_lpole",
                "tolerance_ry",
            ],
        )
        # The stand-in reporter keeps no record of the sphere.
        self.assertIn("domain", record["result"])
        self.assertIsNone(record["result"]["domain"])
        self.assertTrue(any("result.domain" in note for note in record["notes"]))


class DomainDetailsTests(unittest.TestCase):
    record = dict(
        rule_version=1,
        domain_energy_tolerance_ry=1.0e-3,
        radius_from="default rule",
        radius_bohr=31.747411546609168,
        boundary_sphere_radius="16.8 ang",
        hartree_boundary_tolerance="1.0e-03 Ry",
        set_by="atomic densities",
        outermost_atom="H",
        vacuum_angstrom=4.5049,
        wall_estimate_ry=4.3846e-4,
        boundary=dict(about_ry=8.03e-5, bound_ry=2.51e-4, plan="engaged"),
    )

    def test_entries_of_a_record(self) -> None:
        self.assertEqual(domain_details(None), ())
        self.assertEqual(domain_details({}), ())
        self.assertEqual(domain_details(Mock()), ())
        self.assertEqual(domain_details({"after_scf": {"status": "not converged"}}), ())
        self.assertEqual(
            domain_details(self.record),
            (
                (
                    "domain_radius",
                    "31.747412 bohr (16.8 ang; set by atomic densities)",
                ),
                ("domain_vacuum", "4.505 ang beyond the outermost atom (H)"),
                ("domain_energy_tolerance", "1.000000e-03 Ry"),
                ("domain_wall_estimate", "4.384600e-04 Ry"),
                (
                    "domain_boundary_energy",
                    "about 8.030000e-05 Ry, calibrated bound 2.510000e-04 Ry "
                    "(engaged)",
                ),
            ),
        )
        given = dict(
            self.record,
            boundary_sphere_radius=None,
            set_by="input",
            boundary=dict(about_ry=None, bound_ry=None, plan="tolerance off"),
            after_scf=dict(status="not converged"),
        )
        self.assertEqual(
            domain_details(given)[0],
            ("domain_radius", "31.747412 bohr (input; set by input)"),
        )
        self.assertEqual(
            domain_details(given)[-2:],
            (
                (
                    "domain_boundary_energy",
                    "about none, calibrated bound none (tolerance off)",
                ),
                ("domain_wall_after_scf", "none (not converged)"),
            ),
        )
        estimated = dict(
            self.record,
            after_scf=dict(
                status="estimated",
                energy_ry=1.73e-4,
                decay_per_bohr=0.82,
                rough=False,
                radius_limit="",
                boundary_sphere_radius="16.1 ang",
                hartree_boundary_tolerance="6.7e-04 Ry",
            ),
        )
        self.assertEqual(
            domain_details(estimated)[-2:],
            (
                (
                    "domain_wall_after_scf",
                    "1.730000e-04 Ry, decay constant 0.820 /bohr",
                ),
                (
                    "domain_radius_for_wall_share",
                    "16.1 ang with Hartree_Boundary_Tolerance 6.7e-04 Ry",
                ),
            ),
        )
        estimated["after_scf"].update(
            rough=True, radius_limit="at least", hartree_boundary_tolerance=None
        )
        self.assertEqual(
            domain_details(estimated)[-2:],
            (
                (
                    "domain_wall_after_scf",
                    "1.730000e-04 Ry, decay constant 0.820 /bohr, rough",
                ),
                ("domain_radius_for_wall_share", "at least 16.1 ang"),
            ),
        )
        # What held the radius where the density did not, and no radius.
        estimated["after_scf"].update(
            radius_limit="",
            boundary_sphere_radius="34.2 ang",
            hartree_boundary_tolerance="2.0e-04 Ry",
            radius_set_by="multipole order",
        )
        self.assertEqual(
            domain_details(estimated)[-1],
            (
                "domain_radius_for_wall_share",
                "34.2 ang with Hartree_Boundary_Tolerance 2.0e-04 Ry (set by "
                "multipole order)",
            ),
        )
        estimated["after_scf"].update(radius_set_by="density at the sphere")
        self.assertEqual(
            domain_details(estimated)[-1][1],
            "34.2 ang with Hartree_Boundary_Tolerance 2.0e-04 Ry",
        )
        estimated["after_scf"].update(
            boundary_sphere_radius=None,
            hartree_boundary_tolerance=None,
            radius_set_by="none: no radius meets the tolerance",
        )
        self.assertEqual(
            domain_details(estimated)[-1],
            ("domain_radius_for_wall_share", "none: no radius meets the tolerance"),
        )

    def test_finish_adds_the_record_to_the_details_of_the_result(self) -> None:
        messages = []
        translation = parse_parsec_input(DATA / "H_cli_smoke.in")
        reporter = AcceleratedTextReporter(messages.append, translation)
        reporter.reference = Mock()
        reporter.reference.domain = dict(self.record)
        self.assertIs(reporter.domain, reporter.reference.domain)
        result = SimpleNamespace(
            backend=BackendInfo(
                requested="auto", selected="cupy", details=(("hartree_backend", "native"),)
            ),
            backend_statistics=BackendStatistics(),
        )
        reporter.finish(result, 1.0)
        self.assertEqual(
            result.backend.details,
            (("hartree_backend", "native"), *domain_details(self.record)),
        )
        # A box, or a stand-in reporter, adds nothing and leaves the result.
        reporter.reference.domain = None
        untouched = SimpleNamespace(
            backend=SimpleNamespace(selected="cupy", details=()),
            backend_statistics=BackendStatistics(),
        )
        backend = untouched.backend
        reporter.finish(untouched, 1.0)
        self.assertIs(untouched.backend, backend)


def thread_setting(value: str | None):
    """The environment with ``PARSEC_DOMAIN_REPORT_THREAD`` at ``value``, or without it."""

    environment = {
        name: item
        for name, item in os.environ.items()
        if name not in ("PARSEC_DOMAIN_REPORT", "PARSEC_DOMAIN_REPORT_THREAD")
    }
    if value is not None:
        environment["PARSEC_DOMAIN_REPORT_THREAD"] = value
    return patch.dict(os.environ, environment, clear=True)


def radius_threads() -> list[threading.Thread]:
    return [
        thread for thread in threading.enumerate() if thread.name == "parsec-domain-radius"
    ]


class RadiusEstimatesThreadTests(unittest.TestCase):
    """The estimates of the radius search after the SCF, made on a host thread beside it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.potentials = load_pseudopotentials(
            {"H": SpeciesPotential(BASE_PSEUDOPOTENTIALS / "H_POTRE.DAT", 0)}
        )
        # Two hydrogen atoms 40 ang from the centre: the largest multipole
        # order holds the sphere of the rule, 43.9 ang, and a search after
        # the SCF asks for estimates.
        distance = 40.0 * domain.ANGSTROM
        cls.atoms = (Atom("H", (0.0, 0.0, distance)), Atom("H", (0.0, 0.0, -distance)))
        cls.radius = 43.9 * domain.ANGSTROM
        cls.translation = SimpleNamespace(
            domain_energy_tolerance=1.0e-3, boundary_tolerance_from="rule"
        )

    def system(self, selected="cupy", shape="sphere"):
        return SimpleNamespace(
            atoms=self.atoms,
            pseudopotentials=self.potentials,
            electron_count=2.0,
            grid=SimpleNamespace(
                settings=GridSettings(spacing=0.4, radius=self.radius, domain_shape=shape)
            ),
            backend_info=SimpleNamespace(selected=selected),
        )

    def reporter(self, hartree=None, tolerance_from="rule"):
        """A reporter of the smoke input whose reference reads these Hartree settings."""

        reporter = AcceleratedTextReporter(
            lambda _text: None, parse_parsec_input(DATA / "H_cli_smoke.in")
        )
        reporter.reference.problem = SimpleNamespace(
            hartree=HartreeSettings() if hartree is None else hartree
        )
        reporter.reference.translation = SimpleNamespace(
            domain_energy_tolerance=1.0e-3, boundary_tolerance_from=tolerance_from
        )
        self.addCleanup(reporter.close)
        return reporter

    def in_line(self, budget=1.0e-3):
        prepared = domain.PreparedEstimates(
            *valence_point_charges(self.atoms, self.potentials, 2.0)
        )
        domain.prepare_radius_search(prepared, self.radius, 2.0, spacing=0.4, budget=budget)
        return prepared

    def test_switch_is_on_unless_named_off(self) -> None:
        for value, expected in (
            (None, True), ("1", True), ("on", True), ("", True),
            ("0", False), (" OFF ", False), ("false", False), ("no", False),
        ):
            with self.subTest(value=value), thread_setting(value):
                self.assertEqual(domain_report_thread_requested(), expected)

    def test_thread_makes_the_estimates_of_the_preparation_in_line(self) -> None:
        expected = self.in_line()
        self.assertGreater(len(expected), 6)
        before = len(radius_threads())
        worker = RadiusEstimatesThread(self.system(), self.translation)
        (thread,) = radius_threads()[before:]
        self.assertTrue(thread.daemon)
        self.assertIsNot(thread, threading.current_thread())
        # Left alone it ends by itself; stop() then only joins it.
        thread.join(timeout=120)
        self.assertFalse(thread.is_alive())
        prepared = worker.stop()
        self.assertIs(prepared, worker.estimates)
        self.assertIsNone(worker.failure)
        self.assertGreater(worker.seconds, 0.0)
        self.assertEqual(len(radius_threads()), before)
        # The same radii with the same estimates, to the last bit: a search
        # reads from either what it would make itself.
        self.assertEqual(prepared._values, expected._values)
        self.assertEqual(
            [value.hex() for value in prepared._values.values()],
            [value.hex() for value in expected._values.values()],
        )
        self.assertTrue(prepared.holds(expected.positions, expected.charges))
        # The search of the rule after the SCF, from the estimates of the thread.
        fit = domain.WallFit(
            prefactor=1.8e-12, decay=0.9, energy=1.0e-12, rms=0.02, at_scan_end=False, shells=8
        )
        options = dict(spacing=0.4, highest_occupied=-0.5)
        wanted = domain.radius_for_budget(
            fit, self.radius, self.atoms, self.potentials, 2.0, **options
        )
        with patch.object(domain, "estimate_omitted_potential", side_effect=AssertionError):
            read = domain.radius_for_budget(
                fit, self.radius, self.atoms, self.potentials, 2.0, prepared=prepared, **options
            )
        self.assertEqual(read, wanted)
        self.assertEqual((read.radius_text, read.estimates), ("43.9 ang", 6))

    def test_thread_prepares_for_the_energy_tolerance_of_the_input(self) -> None:
        # Domain_Energy_Tolerance of the input, here far below the default:
        # the rule then gives the search a tighter tolerance of the
        # boundary, which larger radii meet, and the search asks for
        # estimates that one with the default tolerance does not.
        budget = 2.0e-6
        translation = SimpleNamespace(
            domain_energy_tolerance=budget, boundary_tolerance_from="rule"
        )
        expected = self.in_line(budget)
        self.assertGreater(len(set(expected._values) - set(self.in_line()._values)), 6)
        worker = RadiusEstimatesThread(self.system(), translation)
        worker._thread.join(timeout=120)
        self.assertFalse(worker._thread.is_alive())
        prepared = worker.stop()
        self.assertIsNone(worker.failure)
        self.assertEqual(
            {radius: value.hex() for radius, value in prepared._values.items()},
            {radius: value.hex() for radius, value in expected._values.items()},
        )
        # The search after the SCF, with that tolerance, makes none itself.
        fit = domain.WallFit(
            prefactor=3.6e-15, decay=0.9, energy=2.0e-15, rms=0.02, at_scan_end=False, shells=8
        )
        options = dict(spacing=0.4, highest_occupied=-0.5, budget=budget)
        wanted = domain.radius_for_budget(
            fit, self.radius, self.atoms, self.potentials, 2.0, **options
        )
        with patch.object(domain, "estimate_omitted_potential", side_effect=AssertionError):
            read = domain.radius_for_budget(
                fit, self.radius, self.atoms, self.potentials, 2.0, prepared=prepared, **options
            )
        self.assertEqual(read, wanted)
        self.assertEqual(
            (read.radius_text, read.tolerance_text, read.estimates),
            ("45.7 ang", "5.6e-05 Ry", 7),
        )
        # A translation that carries no tolerance gets the default.
        worker = RadiusEstimatesThread(self.system(), SimpleNamespace())
        worker._thread.join(timeout=120)
        self.assertEqual(set(worker.stop()._values), set(self.in_line()._values))

    def test_stop_ends_the_thread_behind_the_estimate_it_is_making(self) -> None:
        entered, release, made = threading.Event(), threading.Event(), []
        estimate = domain.estimate_omitted_potential

        def slow(*arguments, **options):
            made.append(arguments[2])
            entered.set()
            self.assertTrue(release.wait(timeout=120))
            return estimate(*arguments, **options)

        with patch.object(domain, "estimate_omitted_potential", slow):
            worker = RadiusEstimatesThread(self.system(), self.translation)
            self.assertTrue(entered.wait(timeout=120))
            stopped = []
            stopper = threading.Thread(target=lambda: stopped.append(worker.stop()))
            stopper.start()
            # stop() waits for the estimate that is being made.
            stopper.join(timeout=0.2)
            self.assertTrue(stopper.is_alive())
            self.assertEqual((len(made), stopped), (1, []))
            release.set()
            stopper.join(timeout=120)
        self.assertFalse(stopper.is_alive())
        # That estimate was finished and kept, and no other was begun.
        self.assertEqual(len(made), 1)
        self.assertEqual(list(stopped[0]._values), made)
        self.assertEqual(radius_threads(), [])
        self.assertIsNone(worker.failure)
        # A second stop finds the thread ended.
        self.assertIs(worker.stop(), stopped[0])

    def test_failure_of_the_thread_is_kept_and_stops_nothing(self) -> None:
        with patch.object(
            accelerated_output, "prepare_radius_search", side_effect=RuntimeError("no estimate")
        ):
            worker = RadiusEstimatesThread(self.system(), self.translation)
            prepared = worker.stop()
        self.assertEqual(worker.failure, "RuntimeError: no estimate")
        self.assertEqual(len(prepared), 0)
        # The search after the SCF makes every estimate it does not find.
        fit = domain.WallFit(
            prefactor=1.8e-12, decay=0.9, energy=1.0e-12, rms=0.02, at_scan_end=False, shells=8
        )
        wanted = domain.radius_for_budget(
            fit, self.radius, self.atoms, self.potentials, 2.0,
            spacing=0.4, highest_occupied=-0.5, prepared=prepared,
        )
        self.assertEqual((wanted.radius_text, wanted.estimates), ("43.9 ang", 6))

    def test_reporter_starts_the_thread_where_the_block_can_ask_for_estimates(self) -> None:
        def started(reporter, system, setting=None, report=True):
            reporter.reference._domain_report = report
            with thread_setting(setting), patch.object(
                accelerated_output, "RadiusEstimatesThread"
            ) as thread:
                reporter._start_radius_estimates(system)
                if thread.call_count:
                    self.assertEqual(thread.call_args.args, (system, reporter.reference.translation))
                    self.assertIs(reporter._radius_thread, thread.return_value)
                else:
                    self.assertIsNone(reporter._radius_thread)
                reporter._radius_thread = None
                return thread.call_count

        # A sphere on the CuPy backend with the tolerance of the rule, or
        # with the default tolerance of the code beside a radius of the input.
        self.assertEqual(started(self.reporter(), self.system()), 1)
        self.assertEqual(started(self.reporter(tolerance_from="default"), self.system()), 1)
        self.assertEqual(started(self.reporter(), self.system(), setting="1"), 1)
        # The switch at its former value, and a run without the block.
        self.assertEqual(started(self.reporter(), self.system(), setting="0"), 0)
        self.assertEqual(started(self.reporter(), self.system(), report=False), 0)
        # A box has no sphere; a tolerance of the input, a tolerance that is
        # off and the direct Coulomb sum ask for no estimate.
        self.assertEqual(started(self.reporter(), self.system(shape="box")), 0)
        self.assertEqual(started(self.reporter(tolerance_from="input"), self.system()), 0)
        self.assertEqual(
            started(self.reporter(HartreeSettings(boundary_tolerance=None)), self.system()), 0
        )
        self.assertEqual(
            started(self.reporter(HartreeSettings(boundary_method="direct")), self.system()), 0
        )
        # The host backends work on the cores the thread would take.
        for selected in ("scipy", "native"):
            self.assertEqual(started(self.reporter(), self.system(selected)), 0)
        # Nothing that goes wrong in starting it stops a calculation.
        reporter = self.reporter()
        with thread_setting(None), patch.object(
            accelerated_output, "RadiusEstimatesThread", side_effect=RuntimeError
        ):
            reporter._start_radius_estimates(self.system())
        self.assertIsNone(reporter._radius_thread)
        reporter._start_radius_estimates(SimpleNamespace())
        self.assertIsNone(reporter._radius_thread)

    def test_reporter_ends_the_thread_before_it_reports_and_on_every_way_out(self) -> None:
        expected = self.in_line()
        reporter = self.reporter()
        with thread_setting(None):
            reporter._start_radius_estimates(self.system())
        worker = reporter._radius_thread
        self.assertIsInstance(worker, RadiusEstimatesThread)
        self.assertIsNone(reporter.reference.prepared_estimates)
        seen = []

        def reference_finish(result, elapsed, *, forces=None):
            # The thread has ended, and the block reads what it made.
            seen.append((len(radius_threads()), reporter.reference.prepared_estimates))

        reporter.reference.finish = reference_finish
        reporter.reference.domain = None
        result = SimpleNamespace(
            backend=SimpleNamespace(selected="cupy", details=()),
            backend_statistics=BackendStatistics(),
        )
        self.assertIsNone(reporter.radius_thread_record)
        reporter.finish(result, 1.0)
        self.assertEqual(seen, [(0, worker.estimates)])
        self.assertIsNone(reporter._radius_thread)
        # What became of the thread stays with the reporter: the estimates
        # it made, its seconds and no failure.
        self.assertEqual(
            reporter.radius_thread_record,
            dict(estimates=len(worker.estimates), seconds=worker.seconds, failure=None),
        )
        self.assertGreater(worker.seconds, 0.0)
        # stop() was called before the thread had ended by itself or after:
        # either way what it kept are estimates of the preparation in line.
        for radius, value in worker.estimates._values.items():
            self.assertEqual(value.hex(), expected.get(radius).hex())
        # close() after the report does nothing, as often as it is called.
        reporter.close()
        reporter.close()
        self.assertIs(reporter.reference.prepared_estimates, worker.estimates)
        # A calculation that does not reach its report: close() ends the thread.
        other = self.reporter()
        with thread_setting(None):
            other._start_radius_estimates(self.system())
        self.assertEqual(len(radius_threads()), 1)
        other.close()
        self.assertEqual(radius_threads(), [])
        self.assertIsNotNone(other.reference.prepared_estimates)
        self.assertEqual(
            (other.radius_thread_record["estimates"], other.radius_thread_record["failure"]),
            (len(other.reference.prepared_estimates), None),
        )
        # A reporter that started none has nothing to end, and no record.
        idle = self.reporter()
        idle.close()
        self.assertIsNone(idle.reference.prepared_estimates)
        self.assertIsNone(idle.radius_thread_record)
        with thread_setting("0"):
            idle._start_radius_estimates(self.system())
        idle.close()
        self.assertIsNone(idle.radius_thread_record)

    def test_reporter_keeps_what_the_thread_raised_and_what_kept_it_from_starting(self) -> None:
        # A thread that fails every time would return every run to the
        # time of the search after the SCF without a sign: the reporter
        # keeps the failure beside the estimates and the seconds.
        failed = self.reporter()
        with thread_setting(None), patch.object(
            accelerated_output, "prepare_radius_search", side_effect=RuntimeError("no estimate")
        ):
            failed._start_radius_estimates(self.system())
            failed.close()
        record = failed.radius_thread_record
        self.assertEqual(
            (record["estimates"], record["failure"], sorted(record)),
            (0, "RuntimeError: no estimate", ["estimates", "failure", "seconds"]),
        )
        self.assertGreaterEqual(record["seconds"], 0.0)
        self.assertEqual(len(failed.reference.prepared_estimates), 0)
        # A thread that could not be started: nothing ran, and the record says why.
        unstarted = self.reporter()
        with thread_setting(None), patch.object(
            accelerated_output, "RadiusEstimatesThread", side_effect=RuntimeError("no thread")
        ):
            unstarted._start_radius_estimates(self.system())
        self.assertEqual(
            unstarted.radius_thread_record,
            dict(estimates=0, seconds=0.0, failure="not started: RuntimeError: no thread"),
        )
        unstarted.close()
        self.assertIsNone(unstarted.reference.prepared_estimates)
        self.assertEqual(unstarted.radius_thread_record["failure"], "not started: RuntimeError: no thread")
        # A thread that is stopped before it has made every estimate
        # records those it made.
        entered, release = threading.Event(), threading.Event()
        estimate = domain.estimate_omitted_potential

        def slow(*arguments, **options):
            entered.set()
            self.assertTrue(release.wait(timeout=120))
            return estimate(*arguments, **options)

        stopped = self.reporter()
        with thread_setting(None), patch.object(domain, "estimate_omitted_potential", slow):
            stopped._start_radius_estimates(self.system())
            worker = stopped._radius_thread
            self.assertTrue(entered.wait(timeout=120))
            closer = threading.Thread(target=stopped.close)
            closer.start()
            # The estimate in flight is let go once the thread was told to end.
            self.assertTrue(worker._ended.wait(timeout=120))
            release.set()
            closer.join(timeout=120)
        self.assertFalse(closer.is_alive())
        self.assertEqual(
            (stopped.radius_thread_record["estimates"], stopped.radius_thread_record["failure"]),
            (1, None),
        )


class AcceleratedCommandLineTests(unittest.TestCase):
    def run_smoke(self, text, *arguments, report=None):
        environment = {
            name: value
            for name, value in os.environ.items()
            if name != "PARSEC_DOMAIN_REPORT" and name not in LAUNCH_BACKENDS
        }
        if report is not None:
            environment["PARSEC_DOMAIN_REPORT"] = report
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(os.environ, environment, clear=True),
        ):
            folder = Path(folder).resolve()
            (folder / "parsec.in").write_bytes(text.encode("utf-8"))
            (folder / "H_POTRE.DAT").write_bytes((DATA / "H_POTRE.DAT").read_bytes())
            output = StringIO()
            with redirect_stdout(output):
                code = cli_module.main(
                    [str(folder / "parsec.in"), "--backend", "scipy", *arguments]
                )
            log = folder / "parsec.out"
            archive = folder / "parsec_python_results.npz"
            details = {}
            if archive.exists():
                with np.load(archive, allow_pickle=False) as data:
                    details = dict(
                        zip(
                            data["backend_detail_keys"].tolist(),
                            data["backend_detail_values"].tolist(),
                        )
                    )
            return (
                code,
                output.getvalue(),
                log.read_text(encoding="utf-8") if log.exists() else "",
                details,
            )

    def test_smoke_runs_leave_the_backends_of_the_launch_out(self) -> None:
        # With the Hartree backend of a GPU launch named, a SciPy smoke run
        # ends with 1 before it prepares anything.
        text = (DATA / "H_cli_smoke.in").read_text(encoding="utf-8")
        launch = {name: "cupy" for name in LAUNCH_BACKENDS[:3]}
        launch["PARSEC_CUPY_RESIDENT_HARTREE"] = "auto"
        with patch.dict(os.environ, launch):
            code, _console, log, _details = self.run_smoke(text, "--quiet")
            self.assertEqual(os.environ["PARSEC_HARTREE_LINEAR_BACKEND"], "cupy")
        self.assertEqual(code, 3)
        self.assertIn(" Selected backend  = scipy", log.split("\n"))

    def test_scf_with_the_radius_left_to_the_rule(self) -> None:
        text = parsed_inputs.without_radius(
            (DATA / "H_cli_smoke.in").read_text(encoding="utf-8")
        )
        text = text.replace("Grid_Spacing 0.8 bohr", "Grid_Spacing 1.2 bohr")
        text = text.replace("Max_Iter: 1", "Max_Iter: 40")
        code, _console, log, details = self.run_smoke(text, "--quiet")
        self.assertEqual(code, 0)
        lines = log.split("\n")
        start = lines.index(" --- Radius is  14.172952 bohrs")
        self.assertEqual(
            lines[start + 1],
            " --- Radius 7.5 ang chosen by the default rule 1 "
            "(Domain_Energy_Tolerance = 1.0E-03 Ry)",
        )
        self.assertIn("       Boundary_Sphere_Radius: 7.5 ang", lines)
        self.assertIn("       Hartree_Boundary_Tolerance: 1.0e-03 Ry", lines)
        self.assertIn(" Selected backend  = scipy", lines)
        self.assertTrue(
            any(
                line.startswith(" --- Energy left by the boundary values at order 9")
                for line in lines
            )
        )
        self.assertEqual(
            sum(line.startswith(" Density at the sphere: the sphere adds") for line in lines),
            1,
        )
        self.assertIn("   For this system the wall share would be met by:", lines)
        # The archive keeps the record in the details of the backend.
        self.assertEqual(
            details["domain_radius"],
            "14.172952 bohr (7.5 ang; set by no atomic density for H)",
        )
        self.assertEqual(
            details["domain_vacuum"], "7.500 ang beyond the outermost atom (H)"
        )
        self.assertEqual(details["domain_energy_tolerance"], "1.000000e-03 Ry")
        self.assertRegex(
            details["domain_wall_after_scf"],
            r"^\d\.\d{6}e-\d\d Ry, decay constant \d\.\d{3} /bohr",
        )
        self.assertRegex(
            details["domain_radius_for_wall_share"],
            r"\d\.\d ang with Hartree_Boundary_Tolerance 1\.0e-03 Ry$",
        )

    def test_input_with_a_radius_only_gains_lines(self) -> None:
        text = (DATA / "H_cli_smoke.in").read_text(encoding="utf-8")
        code, _console, log, details = self.run_smoke(text, "--quiet")
        self.assertEqual(code, 3)
        lines = log.split("\n")
        self.assertIn(" --- Radius is   4.000000 bohrs", lines)
        self.assertFalse(any("default rule" in line for line in lines))
        self.assertIn(
            " Density at the sphere: no estimate, the SCF did not converge", lines
        )
        self.assertEqual(details["domain_radius"], "4.000000 bohr (input; set by input)")
        self.assertEqual(details["domain_wall_after_scf"], "none (not converged)")

    def test_switch_restores_the_details_of_an_input_with_a_radius(self) -> None:
        # PARSEC_DOMAIN_REPORT=0: the archive of an input with a radius has
        # the details it had before the default rule, and its log no line
        # of the sphere.
        text = parsed_inputs.two_atom_smoke()
        code, _console, log, details = self.run_smoke(text, "--quiet")
        self.assertEqual(code, 0)
        added = sorted(name for name in details if name.startswith("domain_"))
        self.assertEqual(
            added,
            [
                "domain_boundary_energy",
                "domain_energy_tolerance",
                "domain_radius",
                "domain_radius_for_wall_share",
                "domain_vacuum",
                "domain_wall_after_scf",
                "domain_wall_estimate",
            ],
        )
        self.assertIn(" Density at the sphere: the sphere adds", log)
        code, _console, quiet_log, former = self.run_smoke(text, "--quiet", report="0")
        self.assertEqual(code, 0)
        self.assertEqual(
            sorted(former), sorted(name for name in details if name not in added)
        )
        for words in ("Density at the sphere", "Sphere of the input", "wall share"):
            self.assertNotIn(words, quiet_log)
        self.assertIn("Self-consistency convergence achieved.", quiet_log.split("\n"))
        code, console, _log, _details = self.run_smoke(text, "--dry-run", report="0")
        self.assertEqual(code, 0)
        self.assertNotIn("default rule", console)
        # A radius of the rule keeps its record; only the block after the
        # SCF goes.
        chosen = parsed_inputs.without_radius(text).replace(
            "Grid_Spacing 0.8 bohr", "Grid_Spacing 1.2 bohr"
        )
        code, _console, log, details = self.run_smoke(chosen, "--quiet", report="off")
        self.assertEqual(code, 0)
        self.assertEqual(
            details["domain_radius"],
            "14.928842 bohr (7.9 ang; set by no atomic density for H)",
        )
        self.assertNotIn("domain_wall_after_scf", details)
        self.assertNotIn("Density at the sphere", log)
        self.assertIn("       Boundary_Sphere_Radius: 7.9 ang", log.split("\n"))

    def test_report_without_a_radius_does_not_lose_the_result(self) -> None:
        # An input with a radius whose tolerance no multipole order meets
        # passed its dry run before the default rule existed.
        text = parsed_inputs.two_atom_smoke("Hartree_Boundary_Tolerance: 1e-200 Ry")
        code, console, log, _details = self.run_smoke(text, "--dry-run")
        self.assertEqual((code, log), (0, ""))
        self.assertIn(
            " would give no radius: no radius meets the Hartree boundary "
            "tolerance of 1.0e-200 Ry at the largest multipole order 60\n",
            console,
        )
        # The search after the SCF gives up for a tolerance of the rule that
        # no order meets: the calculation ends as it should, its archive is
        # written, and the details say that there is no radius.
        text = parsed_inputs.two_atom_smoke("Domain_Energy_Tolerance: 1e-300 Ry")
        code, _console, log, details = self.run_smoke(text, "--quiet")
        self.assertEqual(code, 0)
        lines = log.split("\n")
        self.assertIn("Self-consistency convergence achieved.", lines)
        self.assertTrue(
            any(line.startswith("   No radius for the wall share: ") for line in lines)
        )
        self.assertFalse(any("Calculation failed" in line for line in lines))
        self.assertRegex(
            details["domain_wall_after_scf"], r"^\d\.\d{6}e-\d\d Ry, decay constant"
        )
        self.assertRegex(
            details["domain_radius_for_wall_share"],
            r"^none: no radius meets the Hartree boundary tolerance of ",
        )

    def test_command_line_ends_the_thread_of_its_reporter_on_every_way_out(self) -> None:
        text = parsed_inputs.two_atom_smoke()
        closed = []
        close = AcceleratedTextReporter.close

        def counted(reporter):
            closed.append(reporter)
            return close(reporter)

        with patch.object(AcceleratedTextReporter, "close", counted):
            # A calculation that reports: before the report, and on the way out.
            code, _console, _log, _details = self.run_smoke(text, "--quiet")
            self.assertEqual((code, len(closed), len(set(map(id, closed)))), (0, 2, 1))
            # An SCF that fails or is interrupted, and a preparation that fails.
            for stopped, target, expected in (
                (RuntimeError("stopped"), "run_scf", 1),
                (KeyboardInterrupt(), "run_scf", 130),
                (RuntimeError("stopped"), "prepare_single_point", 1),
            ):
                del closed[:]
                with self.subTest(target=target, code=expected), patch.object(
                    cli_module, target, side_effect=stopped
                ):
                    code, _console, _log, _details = self.run_smoke(text, "--quiet")
                    self.assertEqual((code, len(closed)), (expected, 1))
            # A dry run makes no reporter and has none to end.
            del closed[:]
            code, _console, _log, _details = self.run_smoke(text, "--dry-run")
            self.assertEqual((code, closed), (0, []))
        self.assertEqual(radius_threads(), [])

    @unittest.skipUnless(cupy_available(), "the thread belongs to the CuPy backend")
    def test_block_after_the_scf_is_the_same_with_and_without_the_thread(self) -> None:
        # Two atoms on the GPU: the series bound spares their search every
        # estimate, so the thread ends at once; the block, the archive and
        # the details are those of the search made after the SCF.
        text = parsed_inputs.two_atom_smoke()
        started = []

        class Recorded(RadiusEstimatesThread):
            def __init__(self, *arguments):
                started.append(self)
                super().__init__(*arguments)

        def run(setting):
            del started[:]
            with thread_setting(setting), patch.object(
                accelerated_output, "RadiusEstimatesThread", Recorded
            ), tempfile.TemporaryDirectory() as folder:
                folder = Path(folder).resolve()
                (folder / "parsec.in").write_bytes(text.encode("utf-8"))
                (folder / "H_POTRE.DAT").write_bytes((DATA / "H_POTRE.DAT").read_bytes())
                with redirect_stdout(StringIO()):
                    code = cli_module.main(
                        [str(folder / "parsec.in"), "--backend", "cupy", "--quiet"]
                    )
                with np.load(folder / "parsec_python_results.npz", allow_pickle=False) as data:
                    members = {name: data[name] for name in data.files}
                lines = (folder / "parsec.out").read_text(encoding="utf-8").split("\n")
            first = next(
                index for index, line in enumerate(lines)
                if line.startswith(" Density at the sphere:")
            )
            last = next(index for index in range(first, len(lines)) if not lines[index])
            return code, lines[first:last], members, list(started)

        code, block, members, threads = run(None)
        self.assertEqual(code, 0)
        (worker,) = threads
        self.assertEqual((worker.failure, len(worker.estimates)), (None, 0))
        self.assertIn("   For this system the wall share would be met by:", block)
        code, former_block, former, threads = run("0")
        self.assertEqual((code, threads), (0, []))
        self.assertEqual(block, former_block)
        self.assertEqual(sorted(members), sorted(former))
        times = ("timing", "seconds")
        for name in members:
            if any(word in name for word in times) or name == "backend_detail_values":
                continue
            with self.subTest(member=name):
                self.assertEqual(members[name].tobytes(), former[name].tobytes())
        details, former_details = (
            dict(zip(item["backend_detail_keys"].tolist(), item["backend_detail_values"].tolist()))
            for item in (members, former)
        )
        self.assertEqual(sorted(details), sorted(former_details))
        for name in details:
            if "second" not in name:
                with self.subTest(detail=name):
                    self.assertEqual(details[name], former_details[name])
        self.assertEqual(radius_threads(), [])

    def test_dry_run_says_what_the_rule_gives(self) -> None:
        text = (DATA / "H_cli_smoke.in").read_text(encoding="utf-8")
        code, console, log, _details = self.run_smoke(text, "--dry-run")
        self.assertEqual((code, log), (0, ""))
        self.assertIn(
            " --- Without Boundary_Sphere_Radius the default rule 1 would give "
            "(set by no atomic density for H):\n"
            "       Boundary_Sphere_Radius: 7.5 ang\n"
            "       Hartree_Boundary_Tolerance: 1.0e-03 Ry\n",
            console,
        )
        code, console, log, _details = self.run_smoke(
            parsed_inputs.without_radius(text), "--dry-run"
        )
        self.assertEqual((code, log), (0, ""))
        self.assertIn(
            " --- Radius 7.5 ang chosen by the default rule 1 "
            "(Domain_Energy_Tolerance = 1.0E-03 Ry)\n",
            console,
        )


if __name__ == "__main__":
    unittest.main()
