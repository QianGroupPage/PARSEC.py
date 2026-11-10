"""``Vector_Rng_Seed: -1`` under MPI: every rank takes rank 0's clock seed.

Every rank of the accelerated CLI parses the input itself, and the clock seed is drawn during the parse, so two
ranks whose parse falls into different seconds draw different seeds; :func:`shared_vector_seed` hands rank 0's to
all.  The broadcast of a real MPI launch is checked by ``mpi_vector_seed_check.py`` under srun (two nodes).
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

from parsec_python.Input.parsec_input import parse_parsec_input
from parsec_python.acceleration.cli import shared_vector_seed

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


class _Context:
    """The broadcast of an MPI context of ``size`` ranks whose rank 0 holds ``root``."""

    def __init__(self, size: int, root=None) -> None:
        self.size = size
        self.root = root
        self.calls = 0

    def bcast(self, value, root: int = 0):
        self.calls += 1
        return value if self.root is None else self.root


class SharedVectorSeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, True)
        text = (DATA / "H2_parsec.in").read_text(encoding="utf-8")
        (self.directory / "H_POTRE.DAT").write_bytes((DATA / "H_POTRE.DAT").read_bytes())
        self.path = self.directory / "parsec.in"
        self.path.write_text(text + "\nVector_Rng_Seed: -1\n", encoding="utf-8")

    def _parse_at(self, seconds: float):
        from parsec_python.Input import solver_options

        reader = partial(solver_options.read_solver_options, clock=lambda: seconds)
        with patch("parsec_python.Input.parsec_input.read_solver_options", reader):
            return parse_parsec_input(self.path)

    def test_a_rank_whose_clock_differs_takes_the_seed_of_rank_0(self) -> None:
        root = self._parse_at(1_800_000_000.2)
        other = self._parse_at(1_800_000_001.7)  # its parse ended in the next second
        self.assertNotEqual(root.problem.eigensolver.vector_seed, other.problem.eigensolver.vector_seed)
        seeds = (root.problem.eigensolver.vector_seed, root.problem.eigensolver.random_seed)
        context = _Context(2, root=seeds)
        shared = shared_vector_seed(other, context)
        self.assertEqual(context.calls, 1)
        self.assertEqual(repr(shared.problem), repr(root.problem))
        self.assertEqual(shared.problem.eigensolver.basis_seed, root.problem.eigensolver.basis_seed)
        # Rank 0 keeps its own translation, object for object.
        self.assertIs(shared_vector_seed(root, _Context(2)), root)

    def test_one_rank_and_no_mpi_change_nothing(self) -> None:
        translation = self._parse_at(1_800_000_000.0)
        context = _Context(1, root=(1, 1))
        self.assertIs(shared_vector_seed(translation, context), translation)
        self.assertEqual(context.calls, 0)
        with patch.dict("os.environ", {"PARSEC_MPI": ""}):
            self.assertIs(shared_vector_seed(translation), translation)


if __name__ == "__main__":
    unittest.main()
