"""Canonical launcher for the PARSEC-style Python single-point solver.

The public command selects the accuracy-preserving accelerated runtime by
default.  Scientific reference components remain importable from the same
``parsec_python`` package for inspection and focused testing.
"""

import os
from pathlib import Path
import subprocess
import sys


_SOURCE = Path(__file__).resolve().parents[1]
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))


def _use_project_accelerated_runtime() -> None:
    """Re-execute with the checkout's validated accelerated environment.

    This checkout can contain multiple virtual environments.  ``.venv312``
    owns the validated CuPy/native runtime, so prefer it for a source-tree
    launch unless the caller explicitly requests the active interpreter.
    """

    if __name__ != "__main__" or os.name != "nt":
        return
    if os.environ.get(
        "PARSEC_ACCELERATED_USE_ACTIVE_PYTHON", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}:
        return
    project_root = _SOURCE.parent
    preferred_environment = project_root / ".venv312"
    preferred_python = preferred_environment / "Scripts" / "python.exe"
    if not preferred_python.is_file():
        return
    if Path(sys.executable).resolve() == preferred_python.resolve():
        return
    if os.environ.get("PARSEC_ACCELERATED_RUNTIME_BOOTSTRAPPED") == "1":
        raise SystemExit(
            "Accelerated runtime bootstrap returned to the wrong Python "
            f"interpreter: {sys.executable}"
        )

    environment = os.environ.copy()
    environment["PARSEC_ACCELERATED_RUNTIME_BOOTSTRAPPED"] = "1"
    environment["VIRTUAL_ENV"] = str(preferred_environment)
    environment["PATH"] = (
        str(preferred_python.parent)
        + os.pathsep
        + environment.get("PATH", "")
    )
    print(
        f"Using project accelerated runtime: {preferred_python}",
        flush=True,
    )
    completed = subprocess.run(
        (
            str(preferred_python),
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ),
        env=environment,
        check=False,
    )
    raise SystemExit(completed.returncode)


_use_project_accelerated_runtime()


# Resident client commands intentionally run before NumPy/SciPy/package
# imports.  The short-lived submitter therefore pays only standard-library
# startup; all scientific imports and the CUDA context live in the worker.
if __name__ == "__main__" and any(
    option in sys.argv[1:]
    for option in (
        "--resident",
        "--resident-server",
        "--resident-start",
        "--resident-status",
        "--resident-stop",
    )
):
    from parsec_python.acceleration import resident as _resident

    if "--resident-server" in sys.argv[1:]:
        raise SystemExit(_resident.serve())
    if "--resident-start" in sys.argv[1:]:
        try:
            _resident.start(Path(__file__))
        except RuntimeError as error:
            print(f"Resident error: {error}", file=sys.stderr)
            raise SystemExit(1) from error
        print(_resident.status_text())
        raise SystemExit(0)
    if "--resident-status" in sys.argv[1:]:
        print(_resident.status_text())
        raise SystemExit(0 if _resident.is_running() else 1)
    if "--resident-stop" in sys.argv[1:]:
        stopped = _resident.stop()
        print(
            "Accelerated resident worker stopped."
            if stopped
            else "Accelerated resident worker is not running."
        )
        raise SystemExit(0 if stopped else 1)
    forwarded = [value for value in sys.argv[1:] if value != "--resident"]
    try:
        raise SystemExit(_resident.submit(forwarded, Path(__file__)))
    except RuntimeError as error:
        print(f"Resident error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


try:
    import numpy  # noqa: F401
    import scipy  # noqa: F401
except ModuleNotFoundError as error:
    if error.name in {"numpy", "scipy"}:
        raise SystemExit(
            f"Missing required package {error.name!r} in {sys.executable}.\n"
            "Install the solver dependencies from the repository root with:\n"
            "  python -m pip install -r src/parsec_python/requirements.txt"
        ) from error
    raise


from parsec_python.acceleration.cli import main  # noqa: E402


if __name__ == "__main__":
    if os.environ.get("PARSEC_MEMORY_CENSUS"):
        # A diagnostic: count the device arrays of every thread from the start (acceleration/benchmarks/memory_census.py).
        from parsec_python.acceleration.benchmarks import memory_census

        memory_census.install()
    raise SystemExit(main())
