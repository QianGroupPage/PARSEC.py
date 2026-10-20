"""Structures from machine-learned force fields (UMA, MACE) for parsec_python.

    python src/parsec_python/mlff.py relax calculation/parsec.in --model uma-s-1p2
    python src/parsec_python/mlff.py models
    python src/parsec_python/mlff.py check uma-s-1p2

See ``MLFF/README.md``.  In a DFT run the same relaxation is
``Initial_Structure: mlff`` in ``parsec.in`` or ``--mlff [MODEL]`` on the
command line of ``main.py``.
"""

from pathlib import Path
import sys

_SOURCE = Path(__file__).resolve().parents[1]
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from parsec_python.MLFF.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
