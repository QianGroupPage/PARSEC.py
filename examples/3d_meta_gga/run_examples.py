"""Run the inputs of this example with the accelerated command line, one process per input.

From the repository root (r2SCAN and TPSS are libxc's: set ``PARSEC_LIBXC_LIBRARY``)::

    python examples/3d_meta_gga/run_examples.py --out RUNS            # every input
    python examples/3d_meta_gga/run_examples.py --out RUNS Si_k444/scan
    python examples/3d_meta_gga/run_examples.py --install RUNS        # logs -> parsec_python.out

The runner of ``examples/0d_meta_gga/run_examples.py`` (see there) on the inputs of this example: each input
runs as ``main.py parsec.in --pp-dir examples/3d_meta_gga --no-archive --gpus 1`` in ``RUNS/<mesh>/<functional>/``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("meta_gga_example_runner", HERE.parent / "0d_meta_gga" / "run_examples.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

if __name__ == "__main__":
    raise SystemExit(runner.main(sys.argv[1:], example=HERE, systems=["Si_k222", "Si_k444"],
                                 variants=["pbe", "scan", "r2scan", "tpss"]))
