"""Clean RSDFT entry point.

This file intentionally stays small: it wires together input parsing,
problem setup, backend selection, output-file initialization, and the
actual RSDFT solve implemented in ``rsdft_solver.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from old_architecture.rsdft_backend import select_solver_backend
from old_architecture.rsdft_input import has_diagmeth_override, load_elements, prepare_input_data
from old_architecture.rsdft_models import SolverSettings
from old_architecture.rsdft_output import (
    initialize_output_file,
    print_radius_selection,
    write_rsdft_parameter_output,
)
from old_architecture.rsdft_setup import prepare_system
from old_architecture.rsdft_solver_version2 import run_rsdft_calculation


def _parse_cli_args(argv: list[str]) -> argparse.Namespace:
    """Parse the optional input path and forced backend selection."""
    parser = argparse.ArgumentParser(
        description="Run the version-two legacy RSDFT driver."
    )
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument(
        "--cpu",
        action="store_true",
        help="Force the CPU backend regardless of file/default settings.",
    )
    backend_group.add_argument(
        "--gpu",
        action="store_true",
        help="Force the GPU backend regardless of file/default settings.",
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Optional input file path (.in, .inp, .json, .dat, .mat, .txt).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    """Run one RSDFT job.

    Input:
        argv: Optional CLI arguments. When provided, ``argv[0]`` is treated
            as an input-file path. When omitted, the function uses
            ``sys.argv[1:]``.

    Output:
        Returns ``None``. Side effects are the generated output files and
        printed run diagnostics.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    driver_start_time = time.perf_counter()
    cli_args = _parse_cli_args(argv)
    base_dir = Path(__file__).resolve().parent

    # 1. Load default solver settings and periodic-table metadata.
    settings = SolverSettings()
    if cli_args.gpu:
        settings.use_gpu = 1
    elif cli_args.cpu:
        settings.use_gpu = 0
    settings.normalize()

    elem, n_elements = load_elements(base_dir)
    cli_input_path = cli_args.input_file

    # 2. Gather geometry, charge state, density mode, and optional overrides.
    input_data = prepare_input_data(elem, settings, cli_input_path, base_dir)

    if input_data.density_method in ["ml", "sad_ml_grid"] and not has_diagmeth_override(input_data.settings_overrides):
        settings.diagmeth = 2
        print("No diagonalization override supplied for ML density input; using diagmeth=2 by default.")

    applied = settings.apply_overrides(input_data.settings_overrides)
    if cli_args.gpu:
        settings.use_gpu = 1
    elif cli_args.cpu:
        settings.use_gpu = 0
    settings.normalize()
    if applied:
        print("Applied solver setting overrides:")
        for name, value in applied.items():
            print(f"  {name}: {value}")
    if cli_args.gpu:
        print("Forced GPU backend from command line.")
    elif cli_args.cpu:
        print("Forced CPU backend from command line.")

    # 3. Select CPU/GPU implementations, then derive the simulation domain.
    backend = select_solver_backend(settings)
    print(f"Successfully loaded {len(input_data.atoms)} atomic species.")
    if backend.pseudo_diag_source != backend.label or backend.pseudo_nl_source != backend.label:
        print(
            f"Using {backend.label.upper()} backend for eigensolver/Hartree stages "
            f"with {backend.pseudo_diag_source.upper()} diagonal and "
            f"{backend.pseudo_nl_source.upper()} nonlocal ionic setup."
        )
    else:
        print(f"Using {backend.label.upper()} backend for available solver stages.")

    problem = prepare_system(input_data, settings, elem, n_elements)
    print(f"Atoms {problem.input_data.atoms}")
    print(f"n_atom {problem.input_data.n_atom}")
    print(f"Z_charge {problem.input_data.z_charge}")
    print(f"Grid spacing (h): {problem.h}")
    print(f"Number of eigenvalues (nev): {problem.nev}")
    print(f"Domain: {problem.domain}")
    print_radius_selection(problem.domain)

    # 4. Initialize the log/output files and launch the actual SCF solve.
    initialize_output_file(problem.paths.output_file, backend.label, problem.input_data.density_method)
    write_rsdft_parameter_output(
        problem.paths.output_file,
        problem.nev,
        problem.input_data.atoms,
        problem.input_data.n_atom,
        problem.domain,
        problem.h,
        settings.poldeg,
        settings.fd_order,
    )
    run_rsdft_calculation(problem, elem, n_elements, backend, run_start_time=driver_start_time)


if __name__ == "__main__":
    main()
