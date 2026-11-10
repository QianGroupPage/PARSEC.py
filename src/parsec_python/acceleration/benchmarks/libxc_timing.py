"""Time libxc's CUDA build against the code's own device PBE on the XC evaluation of an SCF iteration.

    PARSEC_LIBXC_LIBRARY=.../libxc.so python -m parsec_python.acceleration.benchmarks.libxc_timing \
        parsec.in [parsec.in ...] [--spec libxc:gga_x_pbe+gga_c_pbe] [--repeat 10] [--json out.json]

For every input (its ``Correlation_Type`` is replaced by PBE) the accelerated
driver prepares the system on the CuPy backend (the symmetry sectors where
the point group allows them, as an SCF run does); its device PBE evaluator
(``device_xc.py``) and the libxc evaluator of ``--spec``
(``device_libxc.py``) on the same grid, orbits and core density then
evaluate the SAD density of the input: unpolarized, and spin-polarized with
the channels at 0.55/0.45 of it.  The first call of each (kernel compilation,
libxc's handle) is timed apart; then the best and the mean of ``--repeat``
calls, with device synchronization.  Also on the full grid (no orbits) with
``--full-grid``.  The XC energies of both are printed: they differ by the
digits of the code's PBE (``V_xc.libxc.OWN_FUNCTIONAL_COUNTERPARTS``).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from time import perf_counter

import numpy as np


def _timed(cp, function, repeat: int):
    cp.cuda.runtime.deviceSynchronize()
    start = perf_counter()
    result = function()
    cp.cuda.runtime.deviceSynchronize()
    first = perf_counter() - start
    times = []
    for _ in range(repeat):
        cp.cuda.runtime.deviceSynchronize()
        start = perf_counter()
        function()
        cp.cuda.runtime.deviceSynchronize()
        times.append(perf_counter() - start)
    return result, first, min(times), float(np.mean(times))


def run(path: str, spec: str, repeat: int, full_grid: bool, pp_dir: str | None = None) -> dict:
    from parsec_python.Input import parse_parsec_input
    from parsec_python.acceleration.backends.cupy import require_cupy
    from parsec_python.acceleration.driver import prepare_single_point
    from parsec_python.acceleration.V_xc.device_libxc import (
        DeviceLibxcXCEvaluator,
        DeviceSpinLibxcXCEvaluator,
    )
    from parsec_python.acceleration.V_xc.device_xc import DeviceSpinXCEvaluator, DeviceXCEvaluator

    cp, _ = require_cupy()
    problem = parse_parsec_input(path, pseudopotential_directory=pp_dir).problem
    problem = replace(problem, scf=replace(problem.scf, xc_functional="pbe", exact_exchange=None))
    started = perf_counter()
    system = prepare_single_point(problem, backend="cupy", symmetry="off" if full_grid else "auto")
    preparation = perf_counter() - started
    own = system.xc_evaluator
    if not isinstance(own, DeviceXCEvaluator):
        raise RuntimeError(f"the run's PBE evaluator is {type(own).__name__}, not the device PBE")
    grid, reducer, device = own.grid, own.reducer, own.device_id
    core = own._core_source
    density = system.initial_density
    if reducer is not None and not isinstance(density, np.ndarray):
        full = np.asarray(reducer.to_full(density), dtype=np.float64)
    else:
        full = np.asarray(density, dtype=np.float64)

    def field(values):
        # The SCF loop hands the evaluators the orbit fields of the sectors (a full array on the full grid).
        return values if reducer is None else reducer.field(reducer.wedge_values(values))

    density = field(full)
    up, down = field(0.55 * full), field(0.45 * full)
    libxc = DeviceLibxcXCEvaluator(spec, grid, reducer=reducer, core_density=core, device_id=device)
    own_spin = DeviceSpinXCEvaluator("pbe", grid, reducer=reducer, core_density=core, device_id=device)
    libxc_spin = DeviceSpinLibxcXCEvaluator(spec, grid, reducer=reducer, core_density=core, device_id=device)
    record = {
        "input": path,
        "spec": spec,
        "grid_points": int(grid.size),
        "values": int(own.size),
        "orbits": reducer is not None,
        "preparation_seconds": preparation,
    }
    for label, function in (
        ("pbe_device", lambda: own(density)),
        ("libxc_device", lambda: libxc(density)),
        ("pbe_device_spin", lambda: own_spin(up, down)),
        ("libxc_device_spin", lambda: libxc_spin(up, down)),
    ):
        result, first, best, mean = _timed(cp, function, repeat)
        record[label] = {"first": first, "best": best, "mean": mean, "energy": float(result.total_energy)}
    record["energy_difference"] = record["libxc_device"]["energy"] - record["pbe_device"]["energy"]
    record["spin_energy_difference"] = record["libxc_device_spin"]["energy"] - record["pbe_device_spin"]["energy"]
    return record


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--spec", default="libxc:gga_x_pbe+gga_c_pbe")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--json", default=None)
    parser.add_argument("--pp-dir", default=None, help="the folder of the pseudopotential files")
    arguments = parser.parse_args(argv)
    records = []
    for path in arguments.inputs:
        record = run(path, arguments.spec, arguments.repeat, arguments.full_grid, arguments.pp_dir)
        records.append(record)
        print(
            f"{path}: {record['grid_points']} points, {record['values']} values"
            f"{' (orbits)' if record['orbits'] else ''}, preparation {record['preparation_seconds']:.1f} s"
        )
        for label in ("pbe_device", "libxc_device", "pbe_device_spin", "libxc_device_spin"):
            entry = record[label]
            print(
                f"  {label:18s} first {1e3 * entry['first']:9.2f} ms  best {1e3 * entry['best']:8.2f} ms  "
                f"mean {1e3 * entry['mean']:8.2f} ms  E_xc {entry['energy']:.12f} Ry"
            )
        print(
            f"  E_xc(libxc) - E_xc(code PBE): {record['energy_difference']:.3e} Ry, spin "
            f"{record['spin_energy_difference']:.3e} Ry"
        )
    if arguments.json:
        with open(arguments.json, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=1)


if __name__ == "__main__":
    main()
