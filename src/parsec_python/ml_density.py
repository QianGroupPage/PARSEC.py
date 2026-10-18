"""Predict ML initial densities and check the model environments.

    python src/parsec_python/ml_density.py models
    python src/parsec_python/ml_density.py check
    python src/parsec_python/ml_density.py predict --input parsec.in
    python src/parsec_python/ml_density.py predict --xyz mol.xyz --vacuum 5 --points 80

Any interpreter with the PARSEC.py requirements runs it; the models run in
their own environments.  Guide: ``src/parsec_python/MLDensity/README.md``.
"""

from pathlib import Path
import sys


_SOURCE = Path(__file__).resolve().parents[1]
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from parsec_python.MLDensity.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
