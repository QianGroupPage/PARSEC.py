"""What a calculation runs on: --device, --gpus, --gpu-ids and --threads with their variables and defaults."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from parsec_python.acceleration import cli as cli_module
from parsec_python.acceleration import resources
from parsec_python.acceleration.backends import selection as selection_module
from parsec_python.acceleration.backends.selection import BackendSelection, resolve_backend
from parsec_python.acceleration.models import BackendUnavailableError
from parsec_python.acceleration.resources import resolve_resources


SMOKE_INPUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "H_cli_smoke.in"


def _cpus(count):
    """A machine on which the process may use ``count`` CPUs."""

    return patch.object(resources, "allowed_cpus", return_value=(count, "affinity of the process"))


class AllowedCpuTests(unittest.TestCase):
    def test_the_affinity_of_the_process_counts_and_a_scheduler_variable_can_only_lower_it(self):
        affinity = patch.object(resources.os, "sched_getaffinity", create=True, return_value=set(range(32)))
        with affinity:
            self.assertEqual(resources.allowed_cpus({}), (32, "affinity of the process"))
            # A site that does not bind its jobs: the task was given 8 of the 32.
            self.assertEqual(resources.allowed_cpus({"SLURM_CPUS_PER_TASK": "8"}), (8, "SLURM_CPUS_PER_TASK"))
            self.assertEqual(resources.allowed_cpus({"NSLOTS": "12", "PBS_NP": "16"}), (12, "NSLOTS"))
            # A variable never raises the count above what the process may use, and nonsense is ignored.
            for value in ("64", "0", "-4", "many", ""):
                with self.subTest(value=value):
                    self.assertEqual(resources.allowed_cpus({"SLURM_CPUS_PER_TASK": value}),
                                     (32, "affinity of the process"))

    def test_without_an_affinity_the_processors_of_the_machine_count(self):
        with patch.object(resources, "os") as stand_in:
            del stand_in.sched_getaffinity
            stand_in.cpu_count.return_value = None
            stand_in.environ = {}
            self.assertEqual(resources.allowed_cpus({}), (1, "processors of the machine"))
            stand_in.cpu_count.return_value = 6
            self.assertEqual(resources.allowed_cpus({}), (6, "processors of the machine"))


class ResolveResourcesTests(unittest.TestCase):
    def test_the_defaults_take_what_is_there_and_set_nothing_but_the_device(self):
        with _cpus(32):
            chosen = resolve_resources(environ={})
        self.assertEqual((chosen.device, chosen.device_source), ("auto", "the default"))
        self.assertEqual((chosen.gpu_setting, chosen.gpu_source, chosen.names_gpus),
                         (None, "all visible, the default", False))
        # The thread count is left to the extension, which keeps four of the CPUs it finds free.
        self.assertEqual((chosen.threads, chosen.thread_plan, chosen.thread_source),
                         (None, 28, "the default: allowed CPUs less 4"))
        self.assertEqual(chosen.environment(), {"PARSEC_DEVICE": "auto"})

    def test_the_variables_stand_behind_the_options_and_an_option_outranks_its_variable(self):
        environ = {"PARSEC_DEVICE": "cpu", "PARSEC_CUPY_DEVICES": "0,2", "OMP_NUM_THREADS": "12"}
        with _cpus(32):
            from_variables = resolve_resources(environ=environ)
            self.assertEqual((from_variables.device, from_variables.device_source), ("cpu", "PARSEC_DEVICE"))
            self.assertEqual((from_variables.gpu_setting, from_variables.gpu_source),
                             (None, "PARSEC_CUPY_DEVICES=0,2"))
            self.assertEqual((from_variables.threads, from_variables.thread_plan, from_variables.thread_source),
                             (None, 12, "OMP_NUM_THREADS=12"))
            named = resolve_resources(device="gpu", gpu_ids="1", threads="6", environ=environ)
        self.assertEqual((named.device, named.device_source), ("gpu", "--device"))
        self.assertEqual(named.environment(),
                         {"PARSEC_DEVICE": "gpu", "PARSEC_CUPY_DEVICES": "1", "OMP_NUM_THREADS": "6"})

    def test_gpus_are_named_by_number_or_by_index_and_naming_them_asks_for_a_gpu(self):
        with _cpus(16):
            two = resolve_resources(gpus="2", environ={})
            self.assertEqual((two.device, two.device_source), ("gpu", "--gpus"))
            self.assertEqual((two.gpu_setting, two.gpu_count, two.gpu_ids, two.gpu_source),
                             ("0,1", 2, None, "--gpus 2"))
            listed = resolve_resources(gpu_ids="3, 0", environ={})
            self.assertEqual((listed.device, listed.device_source), ("gpu", "--gpu-ids"))
            self.assertEqual((listed.gpu_setting, listed.gpu_count, listed.gpu_ids), ("3,0", None, (3, 0)))
            # Every visible GPU, said aloud: a GPU run on all of them, with no number to check.
            every = resolve_resources(gpus="all", environ={})
            self.assertEqual((every.device, every.gpu_setting, every.names_gpus), ("gpu", "auto", False))
            for wrong in (dict(gpus="0"), dict(gpus="-1"), dict(gpus="two"), dict(gpu_ids="0,0"),
                          dict(gpu_ids="a"), dict(gpu_ids=""), dict(gpu_ids="-1"), dict(gpus="1", gpu_ids="0"),
                          dict(gpus="1", device="cpu"), dict(gpu_ids="0", device="cpu")):
                with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                    resolve_resources(environ={}, **wrong)

    def test_a_named_gpu_must_be_among_those_the_process_sees(self):
        with _cpus(16):
            two = resolve_resources(gpus="2", environ={})
            listed = resolve_resources(gpu_ids="0,3", environ={})
            default = resolve_resources(environ={})
        two.check_gpus(2, {})
        listed.check_gpus(4, {})
        default.check_gpus(1, {})
        with self.assertRaisesRegex(ValueError, r"--gpus 2 asks for more GPUs than there are: this process sees 1$"):
            two.check_gpus(1, {})
        with self.assertRaisesRegex(ValueError, r"--gpu-ids 0,3 names a GPU that is not there: this process sees 2 "
                                                r"\(CUDA_VISIBLE_DEVICES=4,5\), numbered from 0"):
            listed.check_gpus(2, {"CUDA_VISIBLE_DEVICES": "4,5"})

    def test_threads_are_a_number_all_or_the_default_of_the_allocation(self):
        with _cpus(32):
            self.assertEqual(resolve_resources(threads="all", environ={}).environment()["OMP_NUM_THREADS"], "32")
            eight = resolve_resources(threads="8", environ={"OMP_NUM_THREADS": "20"})
            self.assertEqual((eight.threads, eight.thread_plan, eight.thread_source), (8, 8, "--threads 8"))
            many = resolve_resources(threads="64", environ={})
            self.assertEqual((many.threads, many.thread_source), (64, "--threads 64, more than the CPUs allowed"))
            listed = resolve_resources(environ={"OMP_NUM_THREADS": "6,2"})
            self.assertEqual((listed.threads, listed.thread_plan), (None, 6))
            for wrong in ("0", "-2", "some", ""):
                with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                    resolve_resources(threads=wrong, environ={})
        # Few CPUs leave one thread, never none.
        with _cpus(3):
            self.assertEqual(resolve_resources(environ={}).thread_plan, 1)
        # Where only a scheduler variable says how few CPUs the task has, the default is taken from it
        # and handed to the extension, which would count the whole machine.
        with patch.object(resources, "allowed_cpus", return_value=(8, "SLURM_CPUS_PER_TASK")):
            task = resolve_resources(environ={"SLURM_CPUS_PER_TASK": "8"})
        self.assertEqual((task.threads, task.thread_plan), (4, 4))
        self.assertEqual(task.environment(), {"PARSEC_DEVICE": "auto", "OMP_NUM_THREADS": "4"})

    def test_an_explicit_backend_names_its_device_and_refuses_the_other(self):
        with _cpus(8):
            for backend, device in (("scipy", "cpu"), ("native", "cpu"), ("cupy", "gpu")):
                with self.subTest(backend=backend):
                    chosen = resolve_resources(backend=backend, environ={})
                    self.assertEqual((chosen.device, chosen.device_source), (device, f"--backend {backend}"))
                    self.assertEqual(resolve_resources(backend=backend, device=device, environ={}).device, device)
                    other = "gpu" if device == "cpu" else "cpu"
                    with self.assertRaises(ValueError):
                        resolve_resources(backend=backend, device=other, environ={})
            with self.assertRaises(ValueError):
                resolve_resources(backend="native", gpus="1", environ={})
            with self.assertRaisesRegex(ValueError, "--device must be auto, gpu or cpu"):
                resolve_resources(device="cuda", environ={})
            with self.assertRaisesRegex(ValueError, "PARSEC_DEVICE must be auto, gpu or cpu"):
                resolve_resources(environ={"PARSEC_DEVICE": "cuda"})

    def test_the_variables_are_set_for_one_calculation_and_put_back(self):
        with _cpus(16):
            chosen = resolve_resources(device="gpu", gpus="2", threads="4", environ={})
        environ = {"PARSEC_CUPY_DEVICES": "0", "OTHER": "kept"}
        with chosen.applied(environ):
            self.assertEqual(environ, {"PARSEC_DEVICE": "gpu", "PARSEC_CUPY_DEVICES": "0,1",
                                       "OMP_NUM_THREADS": "4", "OTHER": "kept"})
        self.assertEqual(environ, {"PARSEC_CUPY_DEVICES": "0", "OTHER": "kept"})
        with self.assertRaises(RuntimeError), chosen.applied(environ):
            raise RuntimeError("a calculation that fails")
        self.assertEqual(environ, {"PARSEC_CUPY_DEVICES": "0", "OTHER": "kept"})


class ReportTests(unittest.TestCase):
    def test_the_report_says_what_ran_and_why(self):
        native = dict(openmp_enabled=True, openmp_max_threads=28, openmp_thread_source="detected_processors_minus_4")
        with _cpus(32):
            default = resolve_resources(environ={})
            named = resolve_resources(device="gpu", gpus="2", threads="8", environ={})
            cpu = resolve_resources(device="cpu", environ={"OMP_NUM_THREADS": "16"})
        self.assertEqual(default.report_lines("cupy", gpus=(0, 1), visible_gpus=2, native=native), [
            " Execution resources:",
            " --------------------",
            " Device            = GPU (auto, the default: a usable GPU was found)",
            " GPUs              = 2 of 2 visible, index 0,1; all visible, the default",
            " CPU threads       = 28 of 32 CPUs allowed (affinity of the process); the default: allowed CPUs less 4",
            "",
        ])
        # The automatic choice says why it left the GPU alone.
        fallback = default.report_lines("native", fallback_reasons=("CuPy skipped: CuPy found no CUDA-capable device.",),
                                        native=native)
        self.assertEqual(fallback[2:4], [
            " Device            = CPU (auto, the default: no usable GPU: CuPy found no CUDA-capable device.)",
            " GPUs              = none"])
        eight = dict(native, openmp_max_threads=8, openmp_thread_source="OMP_NUM_THREADS")
        self.assertEqual(named.report_lines("cupy", gpus=(0, 1), visible_gpus=4, native=eight)[2:5], [
            " Device            = GPU (gpu, --device)",
            " GPUs              = 2 of 4 visible, index 0,1; --gpus 2",
            " CPU threads       = 8 of 32 CPUs allowed (affinity of the process); --threads 8"])
        # A process that had loaded the kernels before the request keeps its threads, and says so.
        self.assertEqual(named.report_lines("cupy", gpus=(0, 1), visible_gpus=4, native=native)[4],
                         " CPU threads       = 28 of 32 CPUs allowed (affinity of the process); --threads 8, "
                         "asked after the kernels were loaded with 28")
        # Without the native extension there is no team to report: the plan stands.
        self.assertEqual(cpu.report_lines("scipy")[2:5], [
            " Device            = CPU (cpu, --device)",
            " GPUs              = none",
            " CPU threads       = 16 (OMP_NUM_THREADS=16); 32 CPUs allowed (affinity of the process); "
            "the C++ kernels are not loaded"])

    def test_the_dry_run_lines_name_the_plan(self):
        with _cpus(32):
            default = resolve_resources(environ={})
            named = resolve_resources(gpu_ids="1", threads="all", environ={})
        self.assertEqual(default.dry_run_lines("cupy", 2), [
            "Device: GPU (requested auto, the default)",
            "GPUs: all visible, the default; 2 visible to this process",
            "CPU threads: 28 (the default: allowed CPUs less 4); 32 CPUs allowed (affinity of the process)"])
        self.assertEqual(default.dry_run_lines("native"), [
            "Device: CPU (requested auto, the default)",
            "CPU threads: 28 (the default: allowed CPUs less 4); 32 CPUs allowed (affinity of the process)"])
        self.assertEqual(named.dry_run_lines("cupy", 2)[:2], [
            "Device: GPU (requested gpu, --gpu-ids)", "GPUs: --gpu-ids 1; 2 visible to this process"])


class DeviceSelectionTests(unittest.TestCase):
    """PARSEC_DEVICE narrows the automatic backend and refuses an explicit one that contradicts it."""

    def setUp(self):
        self.problem = object()

    def resolve(self, backend, device, *, cupy=(True, None), native=(True, None)):
        with (patch.dict(os.environ, {"PARSEC_DEVICE": device}),
              patch.object(selection_module, "_cupy_status", return_value=cupy) as probe,
              patch.object(selection_module, "_native_status", return_value=native)):
            return resolve_backend(backend, self.problem), probe

    def test_a_cpu_run_never_asks_for_the_gpu(self):
        chosen, probe = self.resolve("auto", "cpu")
        self.assertEqual((chosen.selected, chosen.finite_difference_builder, chosen.hartree_backend,
                          chosen.fallback_reasons), ("native", "native", "native", ()))
        probe.assert_not_called()
        chosen, probe = self.resolve("auto", "cpu", native=(False, "not built"))
        self.assertEqual((chosen.selected, chosen.hartree_backend, chosen.fallback_reasons),
                         ("scipy", "scipy", ("native skipped: not built",)))
        probe.assert_not_called()

    def test_a_gpu_run_stops_where_no_gpu_is_usable(self):
        chosen, _probe = self.resolve("auto", "gpu")
        self.assertEqual((chosen.selected, chosen.hartree_backend), ("cupy", "native"))
        chosen, _probe = self.resolve("auto", "gpu", native=(False, "not built"))
        self.assertEqual((chosen.selected, chosen.hartree_backend), ("cupy", "cupy"))
        with self.assertRaisesRegex(BackendUnavailableError,
                                    "a GPU run was requested .* but no GPU is usable: no device"):
            self.resolve("auto", "gpu", cupy=(False, "no device"))

    def test_the_automatic_device_is_what_it_was(self):
        self.assertEqual(self.resolve("auto", "auto")[0].selected, "cupy")
        fallen, _probe = self.resolve("auto", "auto", cupy=(False, "no device"))
        self.assertEqual((fallen.selected, fallen.fallback_reasons), ("native", ("CuPy skipped: no device",)))
        with patch.dict(os.environ):
            os.environ.pop("PARSEC_DEVICE", None)
            with (patch.object(selection_module, "_cupy_status", return_value=(True, None)),
                  patch.object(selection_module, "_native_status", return_value=(True, None))):
                self.assertEqual(resolve_backend("auto", self.problem).selected, "cupy")

    def test_an_explicit_backend_on_the_other_device_is_refused(self):
        for backend, device in (("scipy", "gpu"), ("native", "gpu"), ("cupy", "cpu")):
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "was requested"):
                self.resolve(backend, device)
        self.assertEqual(self.resolve("native", "cpu")[0].selected, "native")
        self.assertEqual(self.resolve("scipy", "auto")[0].selected, "scipy")
        self.assertEqual(self.resolve("cupy", "gpu")[0].selected, "cupy")
        with self.assertRaisesRegex(ValueError, "PARSEC_DEVICE must be auto, gpu or cpu"):
            self.resolve("auto", "cuda")


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *options, selection=None, visible=2):
        selected = selection or BackendSelection(requested="auto", selected="cupy", finite_difference_builder="native",
                                                 hartree_backend="native")
        output, error = StringIO(), StringIO()
        seen = {}

        def resolve(backend, problem):
            seen.update(device=os.environ.get("PARSEC_DEVICE"), gpus=os.environ.get("PARSEC_CUPY_DEVICES"),
                        threads=os.environ.get("OMP_NUM_THREADS"))
            return selected

        with (patch.object(cli_module, "resolve_backend", side_effect=resolve),
              patch("parsec_python.acceleration.backends.cupy.cupy_device_count", return_value=visible),
              _cpus(32), patch.dict(os.environ), redirect_stdout(output), redirect_stderr(error)):
            for name in ("PARSEC_DEVICE", "PARSEC_CUPY_DEVICES", "OMP_NUM_THREADS"):
                os.environ.pop(name, None)
            code = cli_module.main([str(SMOKE_INPUT), "--dry-run", *options])
            left = {name: os.environ.get(name) for name in ("PARSEC_DEVICE", "PARSEC_CUPY_DEVICES", "OMP_NUM_THREADS")}
        return code, output.getvalue(), error.getvalue(), seen, left

    def test_a_dry_run_says_what_the_calculation_would_run_on(self):
        code, report, error, seen, left = self.run_cli()
        self.assertEqual((code, error), (0, ""))
        self.assertIn("Backend: requested=auto, selected=cupy", report)
        self.assertIn("Device: GPU (requested auto, the default)", report)
        self.assertIn("GPUs: all visible, the default; 2 visible to this process", report)
        self.assertIn("CPU threads: 28 (the default: allowed CPUs less 4); 32 CPUs allowed", report)
        self.assertEqual(seen, dict(device="auto", gpus=None, threads=None))
        # Nothing of the request stays in the process.
        self.assertEqual(left, dict(PARSEC_DEVICE=None, PARSEC_CUPY_DEVICES=None, OMP_NUM_THREADS=None))

    def test_the_options_reach_the_solver_through_their_variables(self):
        code, report, error, seen, left = self.run_cli("--device", "gpu", "--gpus", "2", "--threads", "8")
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(seen, dict(device="gpu", gpus="0,1", threads="8"))
        self.assertIn("Device: GPU (requested gpu, --device)", report)
        self.assertIn("GPUs: --gpus 2; 2 visible to this process", report)
        self.assertIn("CPU threads: 8 (--threads 8); 32 CPUs allowed", report)
        self.assertEqual(left, dict(PARSEC_DEVICE=None, PARSEC_CUPY_DEVICES=None, OMP_NUM_THREADS=None))
        cpu = BackendSelection(requested="auto", selected="native", finite_difference_builder="native",
                               hartree_backend="native")
        code, report, error, seen, _left = self.run_cli("--device", "cpu", "--threads", "all", selection=cpu)
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(seen, dict(device="cpu", gpus=None, threads="32"))
        self.assertIn("Device: CPU (requested cpu, --device)", report)
        self.assertNotIn("GPUs:", report)

    def test_what_cannot_be_is_refused_before_anything_runs(self):
        for options, words in ((("--device", "cpu", "--gpus", "1"), "cannot be combined with a CPU run"),
                               (("--device", "gpu", "--backend", "native"), "cannot be combined with a GPU run"),
                               (("--device", "cpu", "--backend", "cupy"), "cannot be combined with --device cpu"),
                               (("--gpus", "0"), "--gpus takes a positive number or all"),
                               (("--threads", "none"), "--threads takes a positive number, all or auto")):
            with self.subTest(options=options):
                code, report, error, seen, _left = self.run_cli(*options)
                self.assertEqual((code, report, seen), (2, "", {}))
                self.assertIn(words, error)
        # More GPUs than the process sees: said with the count that is there.
        code, _report, error, _seen, _left = self.run_cli("--gpus", "4")
        self.assertEqual(code, 2)
        self.assertIn("--gpus 4 asks for more GPUs than there are: this process sees 2", error)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            cli_module.main([str(SMOKE_INPUT), "--gpus", "1", "--gpu-ids", "0", "--dry-run"])

    def test_a_gpu_request_without_cupy_stops_before_the_calculation_writes_a_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "parsec.out"
            for options in (("--device", "gpu"), ("--backend", "cupy")):
                error = StringIO()
                with (self.subTest(options=options), patch.object(cli_module, "_cupy_installed", return_value=False),
                      patch.object(cli_module, "prepare_single_point") as prepare, _cpus(8),
                      redirect_stdout(StringIO()), redirect_stderr(error)):
                    code = cli_module.main([str(SMOKE_INPUT), "--no-archive", "--log", str(log), *options])
                    self.assertEqual(code, 2)
                    self.assertIn("a GPU run was requested", error.getvalue())
                    self.assertIn("CuPy is not installed in this Python environment", error.getvalue())
                    prepare.assert_not_called()
                    self.assertFalse(log.exists())


if __name__ == "__main__":
    unittest.main()
