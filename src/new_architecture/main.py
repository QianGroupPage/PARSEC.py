"""Folder-local launcher for the modular PARSEC-input solver."""

from pathlib import Path
import sys


try:
    import numpy  # noqa: F401
    import scipy  # noqa: F401
except ModuleNotFoundError as error:
    if error.name in {"numpy", "scipy"}:
        raise SystemExit(
            f"Missing required package {error.name!r} in {sys.executable}.\n"
            "Install this folder's dependencies with:\n"
            "  python -m pip install -r requirements.txt"
        ) from error
    raise


# Allow this file to be run directly from inside the package directory while
# retaining normal package imports for modules that use relative imports.
_SOURCE = Path(__file__).resolve().parents[1]
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from new_architecture.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
