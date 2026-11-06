"""Run under ``srun``/``mpirun`` with two or more ranks: ``Vector_Rng_Seed: -1`` gives every rank rank 0's seed.

    srun -N 2 -n 2 python mpi_vector_seed_check.py

Each rank parses the input as the accelerated CLI does, with a clock that differs by 1.5 s per rank (the skew
of a multi-node launch, made certain), then takes the shared seed (:func:`parsec_python.acceleration.cli.
shared_vector_seed`, a real MPI broadcast).  Rank 0 prints the seeds of every rank before and after as JSON;
the exit code is 1 when they differ after.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from functools import partial
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("PARSEC_MPI", "1")

from parsec_python.Input import solver_options
from parsec_python.Input.parsec_input import parse_parsec_input
from parsec_python.acceleration.Parallel import mpi
from parsec_python.acceleration.cli import shared_vector_seed

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


def main() -> int:
    context = mpi.context()
    directory = Path(tempfile.mkdtemp())
    try:
        (directory / "H_POTRE.DAT").write_bytes((DATA / "H_POTRE.DAT").read_bytes())
        text = (DATA / "H2_parsec.in").read_text(encoding="utf-8") + "\nVector_Rng_Seed: -1\n"
        (directory / "parsec.in").write_text(text, encoding="utf-8")
        seconds = 1_800_000_000.25 + 1.5 * context.rank
        reader = partial(solver_options.read_solver_options, clock=lambda: seconds)
        with patch("parsec_python.Input.parsec_input.read_solver_options", reader):
            translation = parse_parsec_input(directory / "parsec.in")
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    before = translation.problem.eigensolver
    after = shared_vector_seed(translation).problem.eigensolver
    rows = context.comm.allgather(
        [context.rank, before.vector_seed, after.vector_seed, after.random_seed, list(after.basis_seed)]
    )
    same = len({(row[2], row[3], tuple(row[4])) for row in rows}) == 1
    differed = len({row[1] for row in rows}) > 1
    if context.is_root:
        print(json.dumps({"ranks": context.size, "rows": rows, "seeds_differed_before": differed, "same_after": same}))
    return 0 if same and differed else 1


if __name__ == "__main__":
    sys.exit(main())
