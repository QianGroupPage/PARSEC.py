"""Compare an LDA+U or TS input on the accelerated driver with the reference solver.

    python -m parsec_python.acceleration.benchmarks.check_corrections INPUT
        [--run reference accelerated] [--backend auto] [--symmetry auto]
        [--forces] [--json PATH]

Each run prepares the system, solves the SCF loop and, with ``--forces``,
the forces, in this process: first calculations, no cache.  The record has
the wall seconds of the three stages, the total energy and its LDA+U and TS
terms, the levels, the forces, the Hirshfeld volumes, the host high-water
mark and, for the accelerated run, the largest device memory in use that a
sampling thread saw on every visible GPU (every 20 ms; it includes the CUDA
context and what other processes hold).  The differences of the two runs
are printed at the end.  The numbers decide nothing: the tests do.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np


class DeviceMemorySampler:
    """The most device memory in use per GPU, sampled by a thread."""

    def __init__(self, interval: float = 0.02) -> None:
        self.interval = float(interval)
        self.peaks: dict[int, int] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        try:
            import cupy as cp

            count = int(cp.cuda.runtime.getDeviceCount())
        except Exception:
            return self
        self.peaks = {device: 0 for device in range(count)}

        def sample():
            while not self._stop.is_set():
                for device in range(count):
                    try:
                        with cp.cuda.Device(device):
                            free, total = cp.cuda.runtime.memGetInfo()
                    except Exception:
                        continue
                    self.peaks[device] = max(self.peaks[device], int(total - free))
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=sample, name="device-memory-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def _host_high_water() -> int | None:
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        from parsec_python.acceleration.refinement import process_memory_high_water

        return process_memory_high_water()


def _record(result, forces, stages: dict) -> dict:
    energies = result.energies
    record = {
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "total": float(energies.total),
        "hubbard": float(getattr(energies, "hubbard", 0.0) or 0.0),
        "dispersion": float(getattr(energies, "dispersion", 0.0) or 0.0),
        "eigenvalues": [float(value) for value in np.asarray(result.eigenvalues)],
        "occupations": [float(value) for value in np.asarray(result.occupations)],
        "seconds": stages,
        "host_high_water_bytes": _host_high_water(),
        "history_total": [float(step.energies.total) for step in result.history],
        "history_residual": [float(step.weighted_residual) for step in result.history],
    }
    if forces is not None:
        record["forces"] = np.asarray(forces.total).tolist()
        for name in ("hubbard", "dispersion"):
            value = getattr(forces, name, None)
            if value is not None:
                record[f"{name}_forces"] = np.asarray(value).tolist()
    dispersion = getattr(result, "dispersion", None)
    if dispersion is not None:
        record["volumes_free"] = np.asarray(dispersion.volumes.free).tolist()
        record["volumes_effective"] = np.asarray(dispersion.volumes.effective).tolist()
    return record


def run_reference(problem, forces_wanted: bool) -> dict:
    from parsec_python.Forces import compute_forces
    from parsec_python.SCF.single_point import prepare_single_point, run_scf

    started = time.perf_counter()
    system = prepare_single_point(problem)
    prepared = time.perf_counter()
    result = run_scf(system)
    solved = time.perf_counter()
    forces = compute_forces(system, result) if forces_wanted else None
    finished = time.perf_counter()
    return _record(
        result,
        forces,
        {"preparation": prepared - started, "scf": solved - prepared, "forces": finished - solved},
    )


def run_accelerated(problem, forces_wanted: bool, backend: str, symmetry: str) -> dict:
    from parsec_python.acceleration import driver

    with DeviceMemorySampler() as sampler:
        started = time.perf_counter()
        system = driver.prepare_single_point(problem, backend=backend, symmetry=symmetry)
        system.materialize_final_wavefunctions = False
        prepared = time.perf_counter()
        result = driver.run_scf(system)
        solved = time.perf_counter()
        forces = driver.compute_forces(system, result) if forces_wanted else None
        finished = time.perf_counter()
    record = _record(
        result,
        forces,
        {"preparation": prepared - started, "scf": solved - prepared, "forces": finished - solved},
    )
    details = dict(system.backend_info.details)
    record["device_peak_bytes"] = {str(key): value for key, value in sampler.peaks.items()}
    record["orbital_symmetry"] = details.get("orbital_symmetry")
    record["sector_devices"] = details.get("orbital_sector_devices")
    device = getattr(system, "hubbard_device", None)
    if device is not None:
        record["hubbard_device_seconds"] = {
            "occupations": device.occupation_seconds,
            "coefficients": device.bind_seconds,
        }
    return record


def ts_timing(problem, json_path) -> int:
    """Seconds of the reference and of the device TS correction of one density."""

    from parsec_python.Energy.dispersion import dispersion_correction
    from parsec_python.SCF.single_point import prepare_single_point
    from parsec_python.acceleration.Energy import device_dispersion_correction

    system = prepare_single_point(problem)
    density = np.asarray(system.initial_density, dtype=np.float64)
    record = {"grid_points": int(system.grid.size), "atoms": len(system.atoms)}
    started = time.perf_counter()
    host = dispersion_correction(system, density)
    record["reference_seconds"] = time.perf_counter() - started
    with DeviceMemorySampler() as sampler:
        for label in ("device_first_seconds", "device_seconds"):
            started = time.perf_counter()
            device = device_dispersion_correction(system, density)
            record[label] = time.perf_counter() - started
    record["device_peak_bytes"] = {str(key): value for key, value in sampler.peaks.items()}
    record["energy_difference"] = float(device.energy - host.energy)
    record["max_force_difference"] = float(np.max(np.abs(device.forces - host.forces)))
    record["max_relative_volume_difference"] = float(
        np.max(np.abs(device.volumes.effective - host.volumes.effective) / host.volumes.effective)
    )
    print(json.dumps(record))
    if json_path:
        Path(json_path).write_text(json.dumps(record, indent=1))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input")
    parser.add_argument("--run", nargs="+", default=["reference", "accelerated"],
                        choices=("reference", "accelerated"))
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--symmetry", default="auto")
    parser.add_argument("--forces", action="store_true")
    parser.add_argument("--json", default=None)
    parser.add_argument("--ts-timing", action="store_true",
                        help="time only the TS correction of the initial density, host against device")
    arguments = parser.parse_args(argv)

    from parsec_python.Input import parse_parsec_input

    problem = parse_parsec_input(Path(arguments.input)).problem
    if arguments.ts_timing:
        return ts_timing(problem, arguments.json)
    out: dict[str, dict] = {}
    for name in arguments.run:
        if name == "reference":
            out[name] = run_reference(problem, arguments.forces)
        else:
            out[name] = run_accelerated(problem, arguments.forces, arguments.backend, arguments.symmetry)
        record = out[name]
        print(
            f"{name:12s} converged={record['converged']} iterations={record['iterations']} "
            f"E={record['total']:.10f} E_U={record['hubbard']:.10f} "
            f"E_vdW={record['dispersion']:.10f} seconds={record['seconds']} "
            f"host_high_water={record['host_high_water_bytes']} "
            f"device_peaks={record.get('device_peak_bytes')} "
            f"symmetry={record.get('orbital_symmetry')}",
            flush=True,
        )
    if len(out) == 2:
        a, b = out["reference"], out["accelerated"]
        print(f"dE_total = {b['total'] - a['total']:.3e}  dE_U = {b['hubbard'] - a['hubbard']:.3e}  "
              f"dE_vdW = {b['dispersion'] - a['dispersion']:.3e}")
        count = min(len(a["eigenvalues"]), len(b["eigenvalues"]))
        occupied = [index for index in range(count) if a["occupations"][index] > 1.0e-8]
        levels = np.abs(np.array(a["eigenvalues"][:count]) - np.array(b["eigenvalues"][:count]))
        print(f"max |d eigenvalue| = {levels.max():.3e} of {count}; occupied {levels[occupied].max():.3e}")
        steps = min(len(a["history_total"]), len(b["history_total"]))
        history = np.abs(np.array(a["history_total"][:steps]) - np.array(b["history_total"][:steps]))
        print("max |d E| per iteration: " + " ".join(f"{value:.1e}" for value in history))
        for key in ("forces", "hubbard_forces", "dispersion_forces"):
            if key in a and key in b:
                print(f"max |d {key}| = {np.max(np.abs(np.array(a[key]) - np.array(b[key]))):.3e} "
                      f"(max |F| {np.max(np.abs(np.array(a[key]))):.3e})")
        if "volumes_effective" in a and "volumes_effective" in b:
            relative = np.abs(np.array(a["volumes_effective"]) - np.array(b["volumes_effective"]))
            print(f"max rel |d V_eff| = {np.max(relative / np.array(a['volumes_effective'])):.3e}")
    if arguments.json:
        Path(arguments.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
