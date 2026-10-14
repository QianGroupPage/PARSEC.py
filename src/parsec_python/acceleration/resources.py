"""What a calculation runs on: CPU or GPU, which GPUs and how many CPU threads.

The command line names the three explicitly (``--device``, ``--gpus`` or
``--gpu-ids``, ``--threads``).  Each has a default and an environment
variable behind it, so that a launch script or a scheduler can set them
without touching the command:

==============  ======================  =======================================
Option          Variable                Default
==============  ======================  =======================================
``--device``    ``PARSEC_DEVICE``       ``auto``: the GPU if one is usable,
                                        the CPU otherwise
``--gpus``      ``PARSEC_CUPY_DEVICES`` every GPU the process can see
``--threads``   ``OMP_NUM_THREADS``     the CPUs the process may use, less four
==============  ======================  =======================================

An option outranks its variable.  Nothing here asks a scheduler for
resources: a job gets its GPUs and CPUs from the allocation, and the
defaults count what the allocation gave the process, not what the machine
has.  GPU indices are those inside ``CUDA_VISIBLE_DEVICES``, which is left
as the scheduler set it.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
import sys
from typing import Iterator, Mapping, MutableMapping


DEVICE_VARIABLE = "PARSEC_DEVICE"
GPU_VARIABLE = "PARSEC_CUPY_DEVICES"
THREAD_VARIABLE = "OMP_NUM_THREADS"
DEVICES = ("auto", "gpu", "cpu")
# The native extension keeps this many of the CPUs it finds free for Python,
# the CUDA driver and the system where no thread count is named.
RESERVED_THREADS = 4
# Variables in which a scheduler gives the CPUs of a task or a job: Slurm,
# PBS (two spellings), Grid Engine and LSF.  The affinity of the process is
# asked first; these only lower it, where a site does not bind its jobs.
_SCHEDULER_CPU_VARIABLES = (
    "SLURM_CPUS_PER_TASK",
    "PBS_NP",
    "NCPUS",
    "NSLOTS",
    "LSB_DJOB_NUMPROC",
)


def device_policy(environ: Mapping[str, str] | None = None) -> str:
    """``auto``, ``gpu`` or ``cpu`` as ``PARSEC_DEVICE`` names it (default ``auto``)."""

    environ = os.environ if environ is None else environ
    value = environ.get(DEVICE_VARIABLE, "auto").strip().lower() or "auto"
    if value not in DEVICES:
        raise ValueError(f"{DEVICE_VARIABLE} must be auto, gpu or cpu, not {value!r}")
    return value


def allowed_cpus(environ: Mapping[str, str] | None = None) -> tuple[int, str]:
    """The CPUs this process may use, and where the number comes from.

    On Linux this is the affinity of the process, which a scheduler that
    binds its jobs sets to the allocation; elsewhere the processors of the
    machine.  A scheduler variable that names fewer lowers it.
    """

    environ = os.environ if environ is None else environ
    if hasattr(os, "sched_getaffinity"):
        count, source = len(os.sched_getaffinity(0)), "affinity of the process"
    else:
        count, source = os.cpu_count() or 1, "processors of the machine"
    for name in _SCHEDULER_CPU_VARIABLES:
        raw = environ.get(name, "").strip()
        if raw.isdigit() and 0 < int(raw) < count:
            count, source = int(raw), name
    return max(1, count), source


def _gpu_ids(text: str, option: str) -> tuple[int, ...]:
    try:
        ids = tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError:
        ids = ()
    if not ids or any(value < 0 for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(
            f"{option} takes GPU indices separated by commas, each once, such as 0,1; not {text!r}"
        )
    return ids


@dataclass(frozen=True)
class Resources:
    """The resources one calculation was asked to run on.

    ``device`` is the request (``auto``, ``gpu`` or ``cpu``) and
    ``device_source`` what made it.  ``gpu_setting`` is the value
    ``PARSEC_CUPY_DEVICES`` takes for the run, or ``None`` to leave it as it
    is; ``gpu_count`` and ``gpu_ids`` say what was named, for the check
    against the GPUs that are there, and ``gpu_source`` says it in words.
    ``threads`` is the value ``OMP_NUM_THREADS`` takes, or ``None`` to leave
    the choice to the native extension, which reads the variable or keeps
    four of its CPUs free; ``thread_plan`` is the number that is expected
    either way (``None`` where the variable holds no number) and
    ``thread_source`` what decides it.  ``cpus`` are the CPUs the process may
    use.
    """

    device: str
    device_source: str
    gpu_setting: str | None
    gpu_source: str
    gpu_count: int | None
    gpu_ids: tuple[int, ...] | None
    threads: int | None
    thread_plan: int | None
    thread_source: str
    cpus: int
    cpus_source: str

    @property
    def names_gpus(self) -> bool:
        """Whether a number or a list of GPUs was named for this run."""

        return self.gpu_count is not None or self.gpu_ids is not None

    def environment(self) -> dict[str, str]:
        """The variables that carry the request to the solver."""

        variables = {DEVICE_VARIABLE: self.device}
        if self.gpu_setting is not None:
            variables[GPU_VARIABLE] = self.gpu_setting
        if self.threads is not None:
            variables[THREAD_VARIABLE] = str(self.threads)
        return variables

    @contextmanager
    def applied(self, environ: MutableMapping[str, str] | None = None) -> Iterator[None]:
        """Set the variables for one calculation and put back what was there."""

        environ = os.environ if environ is None else environ
        variables = self.environment()
        before = {name: environ.get(name) for name in variables}
        environ.update(variables)
        try:
            yield
        finally:
            for name, value in before.items():
                if value is None:
                    environ.pop(name, None)
                else:
                    environ[name] = value

    def check_gpus(self, visible: int, environ: Mapping[str, str] | None = None) -> None:
        """Refuse a number or an index of GPUs that this process does not see."""

        environ = os.environ if environ is None else environ
        masked = environ.get("CUDA_VISIBLE_DEVICES")
        seen = f"this process sees {visible}" + (
            f" (CUDA_VISIBLE_DEVICES={masked})" if masked is not None else ""
        )
        if self.gpu_count is not None and self.gpu_count > visible:
            raise ValueError(f"--gpus {self.gpu_count} asks for more GPUs than there are: {seen}")
        if self.gpu_ids is not None and any(value >= visible for value in self.gpu_ids):
            raise ValueError(
                f"--gpu-ids {','.join(map(str, self.gpu_ids))} names a GPU that is not there: "
                f"{seen}, numbered from 0"
            )

    def _allowed(self) -> str:
        return f"{self.cpus} CPUs allowed ({self.cpus_source})"

    def dry_run_lines(self, selected: str, visible_gpus: int | None = None) -> list[str]:
        """What a calculation of this command would run on (``--dry-run``)."""

        on_gpu = selected == "cupy"
        lines = [
            f"Device: {'GPU' if on_gpu else 'CPU'} (requested {self.device}, {self.device_source})"
        ]
        if on_gpu:
            seen = "" if visible_gpus is None else f"; {visible_gpus} visible to this process"
            lines.append(f"GPUs: {self.gpu_source}{seen}")
        planned = self.thread_source if self.thread_plan is None else (
            f"{self.thread_plan} ({self.thread_source})"
        )
        lines.append(f"CPU threads: {planned}; {self._allowed()}")
        return lines

    def report_lines(
        self,
        selected: str,
        *,
        fallback_reasons: tuple[str, ...] = (),
        gpus: tuple[int, ...] | None = None,
        visible_gpus: int | None = None,
        native: Mapping[str, object] | None = None,
    ) -> list[str]:
        """What the calculation runs on, for ``parsec.out``.

        ``selected`` is the execution backend that was resolved, ``gpus`` the
        devices it was given and ``visible_gpus`` those the process sees;
        ``native`` is the build information of the native extension where it
        is loaded, which holds the thread count that is in force.
        """

        on_gpu = selected == "cupy"
        if self.device == "auto":
            why = "a usable GPU was found" if on_gpu else "no usable GPU"
            skipped = [reason for reason in fallback_reasons if reason.startswith("CuPy skipped")]
            if skipped and not on_gpu:
                why += ": " + skipped[0].split(": ", 1)[-1]
            choice = f"auto, {self.device_source}: {why}"
        else:
            choice = f"{self.device}, {self.device_source}"
        lines = [
            " Execution resources:",
            " --------------------",
            f" Device            = {'GPU' if on_gpu else 'CPU'} ({choice})",
        ]
        if on_gpu and gpus is not None:
            of = "" if visible_gpus is None else f" of {visible_gpus} visible"
            lines.append(
                f" GPUs              = {len(gpus)}{of}, index {','.join(map(str, gpus))}; {self.gpu_source}"
            )
        elif on_gpu:
            lines.append(f" GPUs              = {self.gpu_source}")
        else:
            lines.append(" GPUs              = none")
        if native is not None and native.get("openmp_enabled"):
            running = native.get("openmp_max_threads")
            how = self.thread_source
            if self.threads is not None and running != self.threads:
                # A process that had loaded the kernels before this request
                # (a resident worker) keeps the threads it started with.
                how += f", asked after the kernels were loaded with {running}"
            lines.append(f" CPU threads       = {running} of {self._allowed()}; {how}")
        else:
            planned = self.thread_source if self.thread_plan is None else (
                f"{self.thread_plan} ({self.thread_source})"
            )
            lines.append(
                f" CPU threads       = {planned}; {self._allowed()}; the C++ kernels are not loaded"
            )
        lines.append("")
        return lines


def resolve_resources(
    *,
    device: str | None = None,
    gpus: str | None = None,
    gpu_ids: str | None = None,
    threads: str | None = None,
    backend: str = "auto",
    environ: Mapping[str, str] | None = None,
) -> Resources:
    """Turn the options of a command into the resources of its calculation.

    ``None`` leaves an option to its variable and its default.  Naming GPUs
    asks for a GPU run, so ``--gpus 2`` alone stops where no GPU is usable
    instead of running on the CPU.  A ``ValueError`` says what cannot be
    combined.
    """

    environ = os.environ if environ is None else environ
    backend = str(backend).strip().lower()

    if device is None:
        requested = device_policy(environ)
        source = DEVICE_VARIABLE if environ.get(DEVICE_VARIABLE, "").strip() else "the default"
    else:
        requested = str(device).strip().lower()
        source = "--device"
        if requested not in DEVICES:
            raise ValueError(f"--device must be auto, gpu or cpu, not {device!r}")

    if gpus is not None and gpu_ids is not None:
        raise ValueError("--gpus and --gpu-ids cannot be combined: name a number or the indices")
    count: int | None = None
    ids: tuple[int, ...] | None = None
    setting: str | None = None
    if gpus is not None:
        text = str(gpus).strip().lower()
        if text == "all":
            setting, gpu_source = "auto", "--gpus all"
        elif text.isdigit() and int(text) > 0:
            count = int(text)
            setting = ",".join(map(str, range(count)))
            gpu_source = f"--gpus {count}"
        else:
            raise ValueError(
                f"--gpus takes a positive number or all, not {gpus!r}; --device cpu runs without a GPU"
            )
    elif gpu_ids is not None:
        ids = _gpu_ids(str(gpu_ids), "--gpu-ids")
        setting = ",".join(map(str, ids))
        gpu_source = f"--gpu-ids {setting}"
    else:
        named = environ.get(GPU_VARIABLE, "").strip()
        gpu_source = (
            f"{GPU_VARIABLE}={named}" if named and named.lower() != "auto" else "all visible, the default"
        )
    if gpus is not None or gpu_ids is not None:
        if requested == "cpu":
            raise ValueError("--gpus and --gpu-ids cannot be combined with a CPU run (--device cpu)")
        if device is None and requested == "auto":
            # A number of GPUs is a request for GPUs: stop where there is none.
            requested, source = "gpu", "--gpus" if gpus is not None else "--gpu-ids"

    if backend in {"scipy", "native"}:
        if requested == "gpu":
            raise ValueError(f"--backend {backend} runs on the CPU and cannot be combined with a GPU run")
        if requested == "auto":
            requested, source = "cpu", f"--backend {backend}"
    elif backend == "cupy":
        if requested == "cpu":
            raise ValueError("--backend cupy runs on the GPU and cannot be combined with --device cpu")
        if requested == "auto":
            requested, source = "gpu", "--backend cupy"

    cpus, cpus_source = allowed_cpus(environ)
    thread_count: int | None = None
    text = "auto" if threads is None else str(threads).strip().lower()
    named = environ.get(THREAD_VARIABLE, "").strip()
    if text == "auto":
        if named:
            # The first number of a list is the team of the outer level.
            first = named.split(",")[0].strip()
            thread_plan = int(first) if first.isdigit() and int(first) > 0 else None
            thread_source = f"{THREAD_VARIABLE}={named}"
        else:
            thread_plan = max(1, cpus - RESERVED_THREADS)
            thread_source = f"the default: allowed CPUs less {RESERVED_THREADS}"
            if cpus_source in _SCHEDULER_CPU_VARIABLES:
                # The extension counts the affinity alone: where a scheduler
                # variable names fewer CPUs, the default is taken from it.
                thread_count = thread_plan
    elif text == "all":
        thread_count = thread_plan = cpus
        thread_source = "--threads all"
    elif text.isdigit() and int(text) > 0:
        thread_count = thread_plan = int(text)
        thread_source = f"--threads {thread_count}"
        if thread_count > cpus:
            thread_source += ", more than the CPUs allowed"
    else:
        raise ValueError(f"--threads takes a positive number, all or auto, not {threads!r}")

    return Resources(
        device=requested,
        device_source=source,
        gpu_setting=setting,
        gpu_source=gpu_source,
        gpu_count=count,
        gpu_ids=ids,
        threads=thread_count,
        thread_plan=thread_plan,
        thread_source=thread_source,
        cpus=cpus,
        cpus_source=cpus_source,
    )


def native_thread_information() -> dict[str, object] | None:
    """Build information of the native extension if this process has loaded it."""

    module = sys.modules.get("parsec_accelerated_native")
    if module is None:
        return None
    try:
        return dict(module.build_info())
    except Exception:  # an extension too old to say
        return None


__all__ = [
    "DEVICES",
    "DEVICE_VARIABLE",
    "GPU_VARIABLE",
    "RESERVED_THREADS",
    "Resources",
    "THREAD_VARIABLE",
    "allowed_cpus",
    "device_policy",
    "native_thread_information",
    "resolve_resources",
]
