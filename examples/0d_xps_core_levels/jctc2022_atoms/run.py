#!/usr/bin/env python3
"""Run the cases in ``cases/``, one after the other; finished ones are skipped.

    python run.py                    # everything
    python run.py --only C O         # single elements
    python run.py --cpu-poisson      # Poisson solves on the CPU instead of the GPU
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import xps_cases as xps  # noqa: E402

#: last line of a finished log
FINISHED = "Total accelerated Python wall time"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, default=HERE / "cases")
    parser.add_argument("--only", nargs="+", default=None, help="elements")
    parser.add_argument("--cpu-poisson", action="store_true")
    arguments = parser.parse_args()
    xps.run_queue(arguments.cases, select=lambda entry: not arguments.only or entry["element"] in arguments.only,
                  cpu_poisson=arguments.cpu_poisson, marker=FINISHED)


if __name__ == "__main__":
    main()
