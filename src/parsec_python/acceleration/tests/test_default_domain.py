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
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from parsec_python.Input import parse_parsec_input
from parsec_python.acceleration import cli as cli_module
from parsec_python.acceleration.Output import AcceleratedTextReporter
from parsec_python.acceleration.Output.accelerated_output import domain_details
from parsec_python.acceleration.benchmarks import mpi_full_scf
from parsec_python.acceleration.models import BackendInfo, BackendStatistics
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
