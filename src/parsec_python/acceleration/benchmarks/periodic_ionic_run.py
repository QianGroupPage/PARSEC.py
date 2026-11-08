"""Wall time and device memory of complete runs of the forces, relaxation and dynamics of a periodic cell.

Writes the Si inputs of :mod:`.periodic_force_profile` with a movement block into the output folder
and runs each through the accelerated command line (``main.py``) in this process, with a thread that
samples the used memory of every visible device (``cudaMemGetInfo``) every 20 ms; optionally also
through the reference command line (``reference_main.py``) for the comparison of the trajectories.

    python -m parsec_python.acceleration.benchmarks.periodic_ionic_run OUT case[:movement] ... [--reference]

``movement`` is ``none`` (a single point without forces), ``forces`` (a single point with
forces), ``bfgs`` (three structures) or ``md``
(three steps, ``test_rng``); the default is ``bfgs``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from .periodic_force_profile import CASES, SI_POTENTIAL, _input_text

MOVEMENTS = {
    "none": "",
    "forces": "Skip_force: .false.\n",
    "bfgs": "Minimization: bfgs\nMovement_Num: 2\nForce_Min: 0.0001\n",
    "md": (
        "Molecular_Dynamics: .true.\nStep_num: 2\nTinit: 500.0\nT_Final: 500.0\nT_Step: 0.0\n"
        "Time_Step: 40.0\ntest_rng: .true.\nCooling_Method: stair\n"
    ),
}


class MemorySampler:
    """Largest used memory of every visible device while it runs (bytes, all processes of the device)."""

    def __init__(self, period: float = 0.02) -> None:
        self.period = period
        self.peaks: list[int] = []
        self.baseline: list[int] = []
        self._stop = threading.Event()
        self._thread = None

    def _used(self, cp) -> list[int]:
        used = []
        for device in range(cp.cuda.runtime.getDeviceCount()):
            with cp.cuda.Device(device):
                free, total = cp.cuda.runtime.memGetInfo()
            used.append(int(total - free))
        return used

    def __enter__(self):
        try:
            import cupy as cp
        except Exception:
            return self
        # What an earlier run of this process left in the memory pools is not this run's.
        import gc

        gc.collect()
        for device in range(cp.cuda.runtime.getDeviceCount()):
            with cp.cuda.Device(device):
                cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        self.baseline = self._used(cp)
        self.peaks = list(self.baseline)

        def sample() -> None:
            while not self._stop.is_set():
                for index, value in enumerate(self._used(cp)):
                    self.peaks[index] = max(self.peaks[index], value)
                self._stop.wait(self.period)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def _run(main, path: Path) -> tuple[int, float, str]:
    errors = StringIO()
    started = time.perf_counter()
    with redirect_stdout(StringIO()), redirect_stderr(errors):
        code = main([str(path), "--quiet"])
    return code, time.perf_counter() - started, errors.getvalue()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--reference", action="store_true", help="also the reference command line")
    arguments = parser.parse_args(argv)
    from parsec_python.acceleration.cli import main as accelerated_main
    from parsec_python.cli import main as reference_main

    results = {}
    for item in arguments.cases:
        case, _, movement = item.partition(":")
        movement = movement or "bfgs"
        entry = {}
        runs = [("accelerated", accelerated_main)]
        if arguments.reference:
            runs.append(("reference", reference_main))
        for name, command in runs:
            folder = arguments.out / f"{case}_{movement}_{name}"
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True)
            text = _input_text(**CASES[case]).replace("Skip_force: .false.\n", "") + MOVEMENTS[movement]
            (folder / "parsec.in").write_text(text)
            shutil.copy(SI_POTENTIAL, folder / "Si_POTRE.DAT")
            with MemorySampler() as sampler:
                code, seconds, errors = _run(command, folder / "parsec.in")
            record = {
                "exit": code,
                "wall_seconds": round(seconds, 2),
                "device_peak_gb": [round((p - b) / 1e9, 3) for p, b in zip(sampler.peaks, sampler.baseline)],
                "routed": "reference parsec_python.cli solver" in errors,
            }
            archive = folder / "parsec_python_results.npz"
            if archive.is_file():
                with np.load(archive, allow_pickle=True) as data:
                    for key in ("energy_total_ry", "forces_total", "relaxation_positions", "relaxation_energies",
                                "dynamics_positions", "dynamics_energies"):
                        if key in data.files:
                            record[key] = np.asarray(data[key]).tolist()
            entry[name] = record
            print(case, movement, name, {k: v for k, v in record.items() if not isinstance(v, list)}, flush=True)
        if "reference" in entry:
            differences = {}
            for key in ("energy_total_ry", "forces_total", "relaxation_positions", "relaxation_energies",
                        "dynamics_positions", "dynamics_energies"):
                if key in entry["accelerated"] and key in entry["reference"]:
                    differences[key] = float(
                        np.max(np.abs(np.asarray(entry["accelerated"][key]) - np.asarray(entry["reference"][key])))
                    )
            entry["max_difference"] = differences
            print(case, movement, "accelerated - reference", differences, flush=True)
        results[f"{case}:{movement}"] = entry
    (arguments.out / "runs.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
