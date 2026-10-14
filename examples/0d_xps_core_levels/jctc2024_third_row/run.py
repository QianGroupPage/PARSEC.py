#!/usr/bin/env python3
"""Run the cases in ``cases/``, cheapest first, one after the other.

A case that already has a finished log (``parsec.out`` with the refined total)
is skipped, so the script can be stopped and started again.

    python run.py                    # everything
    python run.py --only H2S SF6     # single molecules (names of paper/structures)
    python run.py --cpu-poisson      # Poisson solves on the CPU instead of the GPU
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import xps_cases as xps  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, default=HERE / "cases")
    parser.add_argument("--only", nargs="+", default=None, help="molecules by structure name")
    parser.add_argument("--largest", type=float, default=None, help="skip cases above this many grid points")
    parser.add_argument("--cpu-poisson", action="store_true")
    arguments = parser.parse_args()
    xps.run_queue(arguments.cases, select=lambda entry: not arguments.only or entry["structure"] in arguments.only,
                  largest=arguments.largest, cpu_poisson=arguments.cpu_poisson)


if __name__ == "__main__":
    main()
